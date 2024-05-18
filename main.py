# Import convention

import pandas as pd
import numpy as np

from typing import List
import regex as re
import ast

import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template


load_figure_template("darkly")



df = pd.read_csv("data/main_dataset.csv")




artists = pd.read_csv("data/artists.csv", index_col=0)
df = pd.read_csv("data/main_dataset.csv", index_col=0)
embeddings = pd.read_csv("data/final_embedding.csv", index_col=0)
playlist_svd = pd.read_csv("data/playlist_svd.csv", index_col=0).values
train_songs = pd.read_csv("data/train_songs.csv", index_col=0)
train_songs["songs"] = train_songs["songs"].apply(ast.literal_eval)


import pickle

with open("models/svd_playlists.pkl", "rb") as f:
    svd_playlists = pickle.load(f)


from annoy import AnnoyIndex


def build_ann(df, filename, dist_metric="angular", fit=True):
    f = df.shape[1]
    if fit:

        t = AnnoyIndex(f, dist_metric)
        for i in range(df.shape[0]):
            t.add_item(i, df.iloc[i].values)
        t.build(30)  # 10 trees
        t.save(filename)
        return t
    else:
        u = AnnoyIndex(f, dist_metric)

        u.load(filename)
        return u


def get_one_hot_vector(songs):
    one_hot_vector = pd.Series(0, index=df.index)
    one_hot_vector.loc[songs] = 1
    return one_hot_vector


u_cb = build_ann(embeddings, "models/cb_annoy.ann", dist_metric="angular", fit=False)

u_cf = build_ann(playlist_svd, "models/cf_annoy.ann", dist_metric="angular", fit=False)


def get_recommendations_cb(track_id, u, df, n=10, how="closest"):
    if type(track_id) == str:
        idx = df.index.get_loc(track_id)
        idxs = u.get_nns_by_item(idx, n + 1)
        recommend_id = df.iloc[idxs].index[1:]
        return recommend_id
    elif type(track_id) == list:
        recommend_idx = []
        idx = [df.index.get_loc(t) for t in track_id]
        for i in idx:
            idxs = u.get_nns_by_item(i, n)
            recommend_idx.extend(idxs)
        if how == "closest":
            recommend_idx = list(set(recommend_idx))
            recommend_idx = [i for i in recommend_idx if i not in idx]
            raw_recommend_distance = [
                [u.get_distance(j, i) for i in recommend_idx] for j in idx
            ]
            raw_recommend_distance = np.array(raw_recommend_distance)
            raw_recommend_distance = raw_recommend_distance.mean(axis=0)
            top_recommend_content = np.array(recommend_idx)[
                np.argsort(raw_recommend_distance)[:n].tolist()
            ].tolist()
        elif how == "freq":
            recommend_idx = [i for i in recommend_idx if i not in idx]

            recommend_count = [recommend_idx.count(i) for i in list(set(recommend_idx))]
            recommend_idx = list(set(recommend_idx))
            top_recommend_content = np.array(recommend_idx)[
                np.argsort(recommend_count)[-n:].tolist()
            ].tolist()
        recommend_id = df.iloc[top_recommend_content].index
        return recommend_id


df_name_artist_id = df[["name", "artists_names"]].reset_index()
df_name_artist_id["name_artist"] = (
    df_name_artist_id["name"]
    + " - "
    + df_name_artist_id["artists_names"].apply(lambda x: ", ".join(ast.literal_eval(x)))
)
df_name_artist_id = df_name_artist_id.drop(columns=["name", "artists_names"])

cols = ['danceability',
'energy', 'key', 'loudness', 'speechiness', 'acousticness',
'instrumentalness', 'liveness', 'valence', 'tempo']

df_standardized =  df[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

from scipy.spatial.distance import cdist

# Calculate the angular distance


def get_recommendations_hybrid(songs, u_cf, final_embedding, train_songs, one_hot_vectors_svd, n=10, how='score', multiplier=1):
    one_hot_songs_test = get_one_hot_vector(songs).values.reshape(1,-1)
    one_hot_songs_test_svd = svd_playlists.transform(one_hot_songs_test)
    idx = u_cf.get_nns_by_vector(one_hot_songs_test_svd[0], n*multiplier)
    all_candidate_songs =  train_songs.iloc[idx]['songs'].explode()
    all_candidate_songs = all_candidate_songs.reset_index().set_index('songs')
    if how == 'freq':
        all_candidate_songs = all_candidate_songs[~all_candidate_songs.index.isin(songs)]
        return pd.Series(all_candidate_songs.index).value_counts().head(n).index.tolist()
    angular_distances = cdist(one_hot_vectors_svd[idx], 
                            one_hot_songs_test_svd, metric='cosine')
    playlist_dist = pd.Series(angular_distances.flatten(),index =  train_songs.iloc[idx].index)
    all_candidate_songs = all_candidate_songs.merge(playlist_dist.map(lambda x: 1/(x+0.0001)).rename('score'), left_on = ['index'], right_index= True, how='left')
    all_candidate_songs = all_candidate_songs.groupby(all_candidate_songs.index).sum()
    all_candidate_songs = all_candidate_songs.sort_values(by='score', ascending=False)
    all_candidate_songs = all_candidate_songs[~all_candidate_songs.index.isin(songs)]
    if how == 'score':
        return all_candidate_songs.head(n).index.tolist()
    if how == 'weighted':
        all_candidate_songs =  train_songs.iloc[idx]['songs'].explode()
        song_distances = cdist(final_embedding.loc[all_candidate_songs], final_embedding.loc[songs], metric='cosine').mean(axis=1)
        song_distances = pd.Series(song_distances, index=all_candidate_songs)

        all_candidate_songs = all_candidate_songs.reset_index().merge(pd.DataFrame(playlist_dist, columns = ['playlist_dist']), left_on='index', right_index=True, how='left')
        final_songlist = all_candidate_songs.groupby('songs').apply(lambda x: x['playlist_dist'].fillna(1).prod())
        final_songlist = (final_songlist) * song_distances.drop_duplicates()
        final_songlist = final_songlist[~final_songlist.index.isin(songs)]
        final_songlist = final_songlist[~final_songlist.index.duplicated()]
        return final_songlist.sort_values(ascending=True).head(n).index.tolist()

def search_df(searchterm: str, df: pd.DataFrame = df_name_artist_id):
    if not searchterm:
        return []
    keywords = searchterm.lower().split()

    # ^(?=.*?p=completed)(?=.*?advancebrain)(?=.*?com_ixxocart).*$
    pattern = "^" + "".join([f"(?=.*{keyword})" for keyword in keywords]) + ".*$"

    # pattern = '^'+'|'.join([f'(?=.*{keyword})' for keyword in keywords])

    result = df[
        df["name_artist"]
        .str.lower()
        .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
    ]
    return result["name_artist"].tolist()[:20]


import dash
from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    Output,
    DashProxy,
    Input,
    MultiplexerTransform,
    html,
    State,
)

from dash import no_update


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# Example app setup
app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc_css],
    transforms=[MultiplexerTransform()],
)

app.layout = html.Div(
    [   dbc.Container([
        dbc.Row(dbc.Col(html.H1("Music Recommender")), className="mb-4"),
        dbc.Stack(
            [
                dbc.Col(
                    [
                        dbc.Stack(
                            [   dbc.Row(html.H2("Your Playlist")),
                                dbc.Stack(
                                    [
                                        html.P("Algorithm: "),
                                        dbc.Select(
                                            id="algorithm-select",
                                            options=[
                                                {
                                                    "label": "Content-based (Avg Dist)",
                                                    "value": "cb",
                                                },
                                                {
                                                    "label": "Collaborative (Frequency)",
                                                    "value": "hybrid_freq",
                                                },
                                                {"label": "Collaborative (Score) ", "value": "hybrid"},
                                                {"label": "Hybrid", "value": "hybrid_weighted"},
                                            ],
                                            value="cb",
                                            persistence=True,
                                        ),
                                        html.P("Number of Recommendations: "),
                                        dbc.Input(
                                            id="n-recommendations",
                                            type="number",
                                            value=10,
                                        ),
                                    ],
                                    direction="horizontal",
                                    gap=2,
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Input(
                                            id="search-input",
                                            type="text",
                                            placeholder="Search songs",
                                            className="w-70",
                                        ),
                                        dbc.Button("Search", id="search-button"),
                                    ],
                                    direction="horizontal",
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Select(
                                            id="song-select",
                                            disabled=True,
                                            placeholder="Select songs...",
                                            className="w-75",
                                        ),
                                        dbc.Button("Add to Playlist", id="add-button"),
                                    ],
                                    direction="horizontal",
                                ),
                                dash_table.DataTable(
                                    id="playlist-table",
                                    data=[],
                                    style_table={
                                        "height": "300px",
                                        "overflowY": "auto",
                                    },
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Button(
                                            "Create Recommendations",
                                            id="recommend-button",
                                            style={"marginRight": "10px"},
                                        ),
                                        dbc.Button("Reset", id="reset-button"),
                                    ],
                                    direction="horizontal",
                                ),
                            ],
                            gap=3,
                        )
                    ], width=6, class_name='dbc align-items-start', 
                ),
                dbc.Col([
                    dbc.Row(html.H2("Recommendations")),
                    dbc.Row(dcc.Graph(id="recommendations-graph", figure={},className="invisible", style={"height": "40vh"})),
                    dbc.Row(dash_table.DataTable(id="recommendations-table", data=[], style_table={"height": "40vh", "overflowY": "auto"})),],
                    width=6, class_name='dbc align-items-start'),
            ],
            direction="horizontal",
            gap=3,
        ),
        dcc.Store(id="store"),
    ],class_name="p-3")
    ]
)


def search_songs(query, df=df_name_artist_id):
    results = search_df(query, df)
    return results


@app.callback(
    Output("song-select", "options"),
    Output("song-select", "disabled"),
    Input("search-button", "n_clicks"),
    State("search-input", "value"),
)
def update_song_options(n_clicks, value):
    if n_clicks:
        return search_songs(value), False
    return [], True


@app.callback(
    Output("playlist-table", "data"),
    Output("store", "data"),
    [Input("add-button", "n_clicks")],
    [State("song-select", "value"), State("store", "data")],
)
def update_playlist(n_clicks, selected_songs, data):
    if n_clicks:
        song_id = df_name_artist_id[df_name_artist_id["name_artist"] == selected_songs][
            "track_uri"
        ].values[0]
        if data is None:
            data = [song_id]
        else:
            data.append(song_id)
            data = list(set(data))
        return [
            {
                "song": df.loc[song_id, "name"],
                "artists": ", ".join(ast.literal_eval(df.loc[song_id, "artists_names"])),
            }
            for song_id in data
        ], data
    return no_update, no_update


@app.callback(
    [
        Output("playlist-table", "data"),
        Output("recommendations-table", "data"),
        Output("store", "data"),
        Output("song-select", "disabled"),
        Output("song-select", "options"),
        Output("search-input", "value"),
        Output("recommendations-graph", "figure"),
        Output("recommendations-graph", "className"),
    ],
    Input("reset-button", "n_clicks"),
)
def reset_tables(n_clicks):
    return [], [], [], True, [], "", {}, "invisible"


@app.callback(
    Output("recommendations-table", "data"),
    Output("recommendations-graph", "figure"),
    Output("recommendations-graph", "className"),
    Input("recommend-button", "n_clicks"),
    State("store", "data"),
    State("algorithm-select", "value"),
    State("n-recommendations", "value"),
)
def update_recommendations(n_clicks, data, algorithm, n):
    if n_clicks and len(data) > 0:
        if algorithm == "cb":
            recommendations = get_recommendations_cb(data, u_cb, df, n, how="closest")
        elif algorithm == "hybrid":
            recommendations = get_recommendations_hybrid(
                data, u_cf, embeddings, train_songs, playlist_svd, n, how="score"
            )
        elif algorithm == "hybrid_weighted":
            recommendations = get_recommendations_hybrid(
                data, u_cf, embeddings, train_songs, playlist_svd, n, how="weighted"
            )
        elif algorithm == "hybrid_freq":
            recommendations = get_recommendations_hybrid(
                data, u_cf, embeddings, train_songs, playlist_svd, n, how="freq"
            )
        results = df.loc[recommendations, ["name", "artists_names"]]
        results["artists_names"] = results["artists_names"].apply(
            lambda x: ", ".join(ast.literal_eval(x))
        )

        original_radar = df_standardized.loc[data, cols].mean()
        recommendations_radar = df_standardized.loc[recommendations, cols].mean()


        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=original_radar,
            theta=cols,
            fill='toself',
            name='Your Playlist'
        ))
        fig.add_trace(go.Scatterpolar(
            r=recommendations_radar,
            theta=cols,
            fill='toself',
            name='Recommendations'
        ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )), 
        showlegend=True
        )

        return results.to_dict("records"), fig, "visible"
    return no_update, {}, "invisible"


# @app.callback(
#     [Output('playlist-table', 'data'), Output('recommendations-table', 'data')],
#     Input('reset-button', 'n_clicks')
# )
# def reset_tables(n_clicks):
#     return [], []

if __name__ == "__main__":
    app.run_server(debug=True)
