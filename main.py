import math
import pandas as pd
import json


class Similarity:
    def __init__(self, user: list, data: list):
        self.user = user
        self.csv_data = data

    def euclidean_distance(self):
        x = self.user
        y = self.csv_data
        return math.sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))

    def manhattan_distance(self):
        x = self.user
        y = self.csv_data
        return sum(abs(a-b) for a, b in zip(x, y))

    def square_rooted(self, x: list):
        return round(math.sqrt(sum(a*a for a in x)), 3)

    def cosine_similarity(self):
        x = self.user
        y = self.csv_data
        numerator = sum(a*b for a, b in zip(x, y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return -1 if denominator <= 0 else round(numerator/float(denominator), 3)


class Data:
    class Element:
        def __init__(self, idx: int):
            self.idx = idx
            self.genres: list = []
            self.themes: list = []
            self.demographics: list = []

        def convert_genres(self, types: str):
            return self.genres.extend(eval(types))

        def convert_themes(self, types: str):
            return self.themes.extend(eval(types))

        def convert_demographics(self, types: str):
            return self.demographics.extend(eval(types))

        def get_n_genres(self):
            return len(self.genres)

    def __init__(self, path: str, n_rows: int = None):
        if path is not None:
            self.df_dataset = self.import_file(path, n_rows)
        self.df_user: pd.DataFrame = None
        self.user_data: list[Data.Element] = []

    def import_user(self, path: str):
        if path is None:
            return False
        cols = ['mal_id', 'mark']
        user = pd.read_csv(
            path,
        )

        # reindex if no 'mark' col.
        user = user.reindex(cols, fill_value=1, axis=1)

        self.df_user = user
        self.__add_weight_user()
        df_user = self.df_user
        df_data = self.df_dataset

        idexes: list = []
        for id in df_user['mal_id'].to_list():
            idexes.extend(df_data.index[df_data['mal_id'] == id].to_list())
        self.user_data = self.__convert_to_element_list(idexes)

    def import_file(self, path: str, n_rows: int = None) -> pd.DataFrame:
        cols = ['mal_id', 'score', 'scored_by',
                'genres', 'themes', 'demographics']
        chunk = 100000
        data = pd.read_csv(
            path, encoding='utf-8',
            nrows=n_rows, usecols=cols,
            chunksize=chunk
        )
        # return df.sort_values(by='mal_id')
        # return pd.concat(data)
        df = pd.concat(data)
        return df

    def print_sample(self, n: int):
        df_user = self.df_user
        df_data = self.df_dataset
        print("df_dataset:", df_data.head(n), sep='\n')
        print("df_user:", df_user.head(n), sep='\n')

    def compare_user_and_dataset(self):
        similarity_data: dict = self.__get_similarity()
        weight_data = self.__weight_similarity(similarity_data)
        self.__build_result_to_dataframe(weight_data)

    def __get_similarity(self,):
        df_dataset = self.df_dataset
        user_id = self._list_id_of_user()
        # convert to list <Element>
        dataset = self.__convert_to_element_list(
            list(self.df_dataset['mal_id'].index))

        similarity_data: dict = {}

        for user in self.user_data:
            dataset_to_value = {}
            n_genres = 0
            n_themes = 0
            n_demographics = 0
            user_value = self.__convert_to_value(user)
            for idx in self.df_dataset.index:
                # matched
                if df_dataset['mal_id'][idx] not in user_id:
                    n_genres = sum(
                        x in user.genres for x in dataset[idx].genres)
                    n_themes = sum(
                        x in user.themes for x in dataset[idx].themes)
                    n_demographics = sum(
                        x in user.demographics for x in dataset[idx].demographics)
                dict_n = {
                    'genres': n_genres,
                    'themes': n_themes,
                    'demographics': n_demographics
                }

                sim = Similarity(user_value, list(dict_n.values()))
                dict_sim = {
                    'manhattan': False,
                    'cosine': False,
                    'euclidean': False
                }
                dict_sim['cosine'] = sim.cosine_similarity()
                dict_sim['euclidean'] = sim.euclidean_distance()
                dict_sim['manhattan'] = sim.manhattan_distance()

                # add to dict
                dataset_to_value[dataset[idx].idx] = {
                    # 'n_count': dict_n,
                    'similarity': dict_sim
                }
            similarity_data[user.idx] = dataset_to_value

        return similarity_data

    def __weight_similarity(self, data: dict[int, dict]):
        result = {}
        df_data = self.df_dataset
        user_idx_in_data = self.__list_element_of_user_idex()

        for idx_data in range(df_data.shape[0]):
            all_element = {}
            for idx_user in user_idx_in_data:
                scaled_dict = self.__scale_by_weight(
                    data[idx_user][idx_data], user_idx_in_data.index(idx_user))
                all_element[idx_user] = scaled_dict
            result[idx_data] = self.__sort_similarity(all_element)[0][1]

        sorted_result = self.__list_of_dict_to_dict(
            self.__sort_similarity(result))
        return sorted_result

    def __scale_by_weight(self, data: dict[str, dict], idx_in_user: int):
        df_user = self.df_user
        weight = df_user['weight'][idx_in_user]
        cosine = data['similarity']['cosine']/weight
        manhattan = data['similarity']['manhattan']*weight
        euclidean = data['similarity']['euclidean']*weight
        scaled_element = {
            'similarity': {
                'cosine': cosine,
                'euclidean': euclidean,
                'manhattan': manhattan,
            }
        }
        return scaled_element

    def __add_weight_user(self):
        df_user = self.df_user
        total_mark = df_user['mark'].sum()
        weight = 1/total_mark

        # Add column "weight"
        df_user['weight'] = df_user.apply(lambda x: x['mark']*weight, axis=1)

    def __sort_similarity(self, data: dict) -> list[tuple]:
        def criteria(x): return (
            x[1]['similarity']['cosine'],
            x[1]['similarity']['euclidean'],
            x[1]['similarity']['manhattan'],
        )

        def criteria1(x): return (
            x[1]['similarity']['manhattan'],
        )

        def criteria2(x): return (
            x[1]['similarity']['euclidean'],
        )

        def criteria3(x): return (
            x[1]['similarity']['cosine'],
        )
        ls_sorted = sorted(data.items(), key=criteria1, reverse=False)
        ls_sorted = sorted(ls_sorted, key=criteria2, reverse=False)
        ls_sorted = sorted(ls_sorted, key=criteria3, reverse=True)
        return ls_sorted

    def __list_of_dict_to_dict(self, list: list):
        dict = {}
        for item in list:
            mal_id = item[0]
            dict[mal_id] = item[1]
        return dict

    def _list_id_of_user(self, df_user: pd.DataFrame = None):
        return list(self.df_user['mal_id']) if df_user is None else list(df_user['mal_id'])

    def __convert_to_element_list(self, idexes: list) -> list['Data.Element']:
        return [self.__convert_to_element(idx) for idx in idexes]

    def __convert_to_element(self, idex) -> 'Data.Element':
        df = self.df_dataset
        row = Data.Element(idex)
        row.convert_genres(df['genres'][idex])
        row.convert_themes(df['themes'][idex])
        row.convert_demographics(df['demographics'][idex])
        return row

    def __convert_to_value(self, ele: Element) -> list[int]:
        return [len(ele.genres),
                len(ele.themes),
                len(ele.demographics), ]

    def __build_result_to_dataframe(self, dict: dict[int, dict]):
        df_result = pd.DataFrame.from_dict(
            {(i, j): dict[i][j]
                for i in dict.keys()
                for j in dict[i].keys()},
            orient='index'
        )
        # Delete unnamed col
        df_result = df_result.reset_index(level=1, drop=True)
        # Add 'id' col
        df_result['mal_id'] = self.df_dataset['mal_id']
        df_result.insert(0, 'mal_id', df_result.pop('mal_id'))
        # Filter don't match user
        df_result = df_result[~df_result['mal_id'].isin(
            self.df_user['mal_id'])]
        self.df_result = df_result

    def export_result(
        self, path: str = None, n_rows: int = None,
        sort_index: bool = False, index: bool = True,
        columns: list[str] = None,
    ):
        path = 'result.csv' if path is None else path
        self._export_dataframe_to_csv(
            df=self.df_result, path=path,
            n_rows=n_rows, sort_index=sort_index,
            index=index, columns=columns,
        )

    def _export_dataframe_to_csv(
        self, df: pd.DataFrame, path: str = None,
        columns: list[str] = None, n_rows: int = None,
        sort_index: bool = False, index: bool = True,
    ):
        using_df = df.head(n_rows) if n_rows is not None else df
        using_df = using_df.sort_index() if sort_index else using_df
        using_df.to_csv(path_or_buf=path, columns=columns, index=index)

    def __list_element_of_user_idex(self):
        return [ele.idx for ele in self.user_data]

    def print_json(self, data):
        print(json.dumps(data, indent=2), end='\n\n')


if __name__ == '__main__':
    data = Data("manga.csv")
    data.import_user('user.csv')
    data.compare_user_and_dataset()
    data.export_result(columns=['mal_id'], index=False)
