import pandas

from pathlib import Path
from surprise import Reader, Dataset


DATA_DIRECTORY = Path("data") / "raw"
MOVIES_DIRECTORY = DATA_DIRECTORY / "movies.csv"
RATINGS_DIRECTORY = DATA_DIRECTORY / "ratings.csv"


def load_data_for_training():
    df_mov = pandas.read_csv(MOVIES_DIRECTORY)
    df_rat = pandas.read_csv(RATINGS_DIRECTORY)

    mov_columns = ["movieId", "title"]
    rat_columns = ["userId", "movieId", "rating"]
    
    df_mov = df_mov[mov_columns]
    df_rat = df_rat[rat_columns]

    movie_map = df_mov.set_index("movieId")

    reader = Reader(rating_scale=(1, 5))

    full_data = Dataset.load_from_df(
        df_rat, reader
    )

    trainset = full_data.build_full_trainset()

    print(f"Data loaded. Total ratings: {trainset.n_ratings}")
    print(f"Total unique users: {trainset.n_users}")
    print(f"Total unique items: {trainset.n_items}")
    
    return trainset, full_data, movie_map

trainset, full_data, movie_map = load_data_for_training()
print(trainset)


def get_inner_id_map(trainset):
    return {
        trainset.to_raw_iid(inner_id): inner_id for inner_id in trainset.all_items()
    }