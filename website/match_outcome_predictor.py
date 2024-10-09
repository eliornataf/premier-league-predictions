import os
import numpy as np
import datetime
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from django.db.models import QuerySet, IntegerField
from lightgbm import LGBMClassifier
from pandas.core.groupby import DataFrameGroupBy
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from typing import Tuple, List, Union, Optional, Callable, Dict, Any
from xgboost import XGBClassifier
from website.models import Fixture


class MatchOutcomePredictor:
    """
    A class for creating match outcome predictors based on provided data.

    Attributes:
        _SEASON_END_YEAR (int): The ending year of the season.
        _LAST_GAMES_LIST (List[int]): List of integers representing the number of last games to consider for feature
            engineering.
        _INPUT_DATA_PATH (str): Absolute path to the input data file (row_dataframe.csv).
        _SAVED_MODEL_DIRECTORY (str): Absolute path to the directory for saving models and processed data.
        _PROMOTED_CLUBS_2023_24 (List[str]): List of clubs promoted in the 2023-2024 season.
        _PROMOTED_CLUBS_2024_25 (List[str]): List of clubs promoted in the 2024-2025 season.


    Methods:
        create_match_outcome_predictor(): Creates a match outcome predictor using the provided data and saves the
            processed DataFrame and model.
        update_predicted_fixtures(): Updates the predicted outcomes for fixtures in the database.
        combine_and_save_dataframes(): Processes the list of processed games, converts them to a DataFrame, and appends
            them to an existing CSV file.

    """

    _SEASON_END_YEAR: int = 2025
    _LAST_GAMES_LIST: List[int] = [1, 5, 20, 80, 320, 640, 1080]
    _INPUT_DATA_PATH: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model_data", "row_dataframe.csv")
    )
    _SAVED_MODEL_DIRECTORY: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model_data")
    )
    _PROMOTED_CLUBS_2023_24: List[str] = ["Burnley", "Luton Town", "Sheffield United"]
    _PROMOTED_CLUBS_2024_25: List[str] = [
        "Ipswich Town",
        "Leicester City",
        "Southampton",
    ]

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        """
        Validate a DataFrame based on specific criteria for each column.

        :param dataframe: The DataFrame to be validated.
        :return: None

        :raises ValueError: If any row does not pass validation.
        """
        for index, row in dataframe.iterrows():
            # Validate season_end_year
            if not (
                isinstance(row["season_end_year"], int)
                and 1993
                <= row["season_end_year"]
                <= MatchOutcomePredictor._SEASON_END_YEAR
            ):
                raise ValueError(
                    f"Invalid value '{row['season_end_year']}' for 'season_end_year' in row {index}. "
                    f"It must be an integer between 1993 and {MatchOutcomePredictor._SEASON_END_YEAR}."
                )

            # Validate date
            if not isinstance(row["date"], str):
                raise ValueError(
                    f"Invalid data type '{type(row['date'])}' for 'date' in row {index}. It must be a "
                    f"string."
                )
            try:
                date: datetime.date = pd.to_datetime(row["date"], dayfirst=True).date()
                if not (1992 <= date.year <= MatchOutcomePredictor._SEASON_END_YEAR):
                    raise ValueError(
                        f"Invalid date value '{row['date']}' in row {index}. It must be between 1992 "
                        f"and {MatchOutcomePredictor._SEASON_END_YEAR}."
                    )
            except ValueError:
                raise ValueError(
                    f"Invalid date format '{row['date']}' in row {index}. It must be in the format "
                    f"'YYYY-MM-DD'."
                )

            # Validate home_team and away_team
            if not (
                isinstance(row["home_team"], str)
                and all(char.isalpha() or char.isspace() for char in row["home_team"])
                and isinstance(row["away_team"], str)
                and all(char.isalpha() or char.isspace() for char in row["away_team"])
            ):
                raise ValueError(
                    f"Invalid value '{row['home_team']}' or '{row['away_team']}' for 'home_team' or "
                    f"'away_team' in row {index}. They must be non-empty strings containing alphabetic "
                    f"characters only."
                )

            # Validate full_time_result
            if "full_time_result" in row:
                if not (
                    isinstance(row["full_time_result"], str)
                    and row["full_time_result"] in ("H", "D", "A")
                ):
                    raise ValueError(
                        f"Invalid value '{row['full_time_result']}' for 'full_time_result' in row "
                        f"{index}. It must be one of 'H', 'D', or 'A'."
                    )

            # Validate half_time_result
            if "half_time_result" in row:
                if not (
                    isinstance(row["half_time_result"], str)
                    or pd.isna(row["half_time_result"])
                    or row["half_time_result"] in ("H", "D", "A")
                ):
                    raise ValueError(
                        f"Invalid value '{row['half_time_result']}' for 'half_time_result' in row "
                        f"{index}. It must be one of 'H', 'D', 'A', or NaN."
                    )

            # Validate 'home_team_full_time_goals' and 'away_team_full_time_goals' without NaN
            for col in ["home_team_full_time_goals", "away_team_full_time_goals"]:
                value = row[col]
                if not (str(value).isdigit() and int(value) >= 0):
                    raise ValueError(
                        f"Invalid numeric value '{value}' in column {col} of row {index}. It must be a non-negative "
                        f"integer."
                    )

            numeric_columns_possible_nan: List[str] = [
                "home_team_half_time_goals",
                "away_team_half_time_goals",
                "home_team_shots",
                "away_team_shots",
                "home_team_shots_on_target",
                "away_team_shots_on_target",
                "home_team_corners",
                "away_team_corners",
                "home_team_fouls",
                "away_team_fouls",
                "home_team_yellow_cards",
                "away_team_yellow_cards",
                "home_team_red_cards",
                "away_team_red_cards",
            ]

            # Validate numeric columns with possible NaN values
            for col in numeric_columns_possible_nan:
                value = row[col]
                if not (
                    pd.isna(value) or (isinstance(value, (int, float)) and value >= 0)
                ):
                    raise ValueError(
                        f"Invalid numeric value '{value}' in column {col} of row {index}. It must be a non-negative "
                        f"number or NaN."
                    )

    @staticmethod
    def _get_season(date: pd.Timestamp) -> int:
        """
        Determines the season based on the provided date.

        :param date: The date.
        :return: Numeric representation of the season corresponding to the provided date.
                 Possible values: 0 for 'winter', 1 for 'spring', 2 for 'summer', 3 for 'autumn'.
        """
        # Extract the month from the provided date
        month: int = date.month

        # Determine the season based on the month
        if month in [1, 2, 12]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        elif month in [9, 10, 11]:
            return 3  # autumn

    @staticmethod
    def _add_matchweek_column(dataframe: pd.DataFrame) -> None:
        """
        Add the 'matchweek' column to the dataframe based on the number of clubs and matchweeks for each season.
        For seasons 1993, 1994, and 1995, the number of clubs was 22 and the number of matchweeks was 42. After 1995,
        the number of clubs is 20 and the number of matchweeks is 38.

        :param dataframe: The input dataframe containing 'Date', 'season_end_year', and 'HomeTeam' columns.
        :return: None
        """
        # List to store matchweek values
        matchweeks_list: List[int] = []
        # Initialize the matchweek counter
        current_matchweek: int = 1

        for index, row in dataframe.iterrows():
            # Extract season and number of clubs based on the season end year
            season_end_year: int = row["season_end_year"]
            number_of_clubs: int = 22 if season_end_year in [1993, 1994, 1995] else 20
            number_of_games_per_matchweek: int = number_of_clubs // 2
            number_of_matchweeks: int = (
                42 if season_end_year in [1993, 1994, 1995] else 38
            )

            # Calculate the matchweek based on the Date and the number of games per matchweek
            matchweek: int = current_matchweek
            matchweeks_list.append(matchweek)

            # Increment the current_matchweek after 'number_of_games_per_matchweek' rows
            if season_end_year <= 1995:
                if (index + 1) % number_of_games_per_matchweek == 0:
                    current_matchweek += 1
            else:
                if (index - 5) % number_of_games_per_matchweek == 0:
                    current_matchweek += 1

            # Reset current_matchweek when it reaches the maximum number of matchweeks
            if current_matchweek > number_of_matchweeks:
                current_matchweek = 1

        # Add the matchweek column to the dataframe
        dataframe["matchweek"] = matchweeks_list

    @staticmethod
    def _calculate_recent_form(
        teams_group: DataFrameGroupBy,
        dataframe: pd.DataFrame,
        data_column: str,
        update_column: str,
        last_games: int,
    ) -> None:
        """
        Calculate the recent form (last 'last_games' games) for each team and update the specified column in the given
        dataframe.

        :param teams_group: The grouped dataframe containing teams' data.
        :param dataframe: The original dataframe where the 'update_column' will be updated.
        :param data_column: The column name to use after getting the group from teams_group (data to calculate the
        recent form).
        :param update_column: The column name to insert in the dataframe after calculating the recent form.
        :param last_games: The number of last games to consider for calculating recent form.
        :return: None
        """
        for team_name, team_indices in teams_group.groups.items():
            # Get the last 'last_games' games for each team from the specified data_column
            team_last_games: pd.Series = (
                teams_group.get_group(team_name)[data_column]
                .shift()
                .rolling(last_games, min_periods=last_games)
                .sum()
            )

            # Update the specified column in the dataframe
            dataframe.loc[team_indices, update_column] = team_last_games

    @staticmethod
    def _calculate_team_points(
        team_type: str, result: Union[str, float]
    ) -> Union[int, float]:
        """
        Calculate the points based on the given team type and match result.

        :param team_type: The type of the team, can be 'H' for home team or 'A' for away team.
        :param result: The result of the match, can be 'H' for home team win, 'D' for draw, 'A' for away team win, or
        NumPy NaN.
        :return: The points earned based on the result. 3 points for a win, 1 point for a draw, 0 points for a loss, or
        the input NumPy NaN if the result is NaN.
        """
        if isinstance(result, float) and np.isnan(result):
            return result  # If result is already NumPy NaN, return it.

        if result == team_type:
            return 3
        elif result == "D":
            return 1
        else:
            return 0

    def _calculate_and_assign_team_points(self, dataframe: pd.DataFrame) -> None:
        """
        Calculate and assign home and away team points based on 'full_time' and 'half_time' data columns.

        :param dataframe: The DataFrame to be processed.
        """
        for data_column_suffix in ["full_time", "half_time"]:
            home_data_column: str = f"home_team_{data_column_suffix}_game_points"
            away_data_column: str = f"away_team_{data_column_suffix}_game_points"

            dataframe[home_data_column] = dataframe.apply(
                lambda row: self._calculate_team_points(
                    "H", row[f"{data_column_suffix}_result"]
                ),
                axis=1,
            )
            dataframe[away_data_column] = dataframe.apply(
                lambda row: self._calculate_team_points(
                    "A", row[f"{data_column_suffix}_result"]
                ),
                axis=1,
            )

    def _calculate_and_update_recent_form_points(
        self,
        dataframe: pd.DataFrame,
        last_games: int,
        data_column_suffix: str,
        home_teams_group: DataFrameGroupBy,
        away_teams_group: DataFrameGroupBy,
        form_type: str,
        swap_home_away: bool,
    ) -> None:
        """
        Calculate and update the recent form points (last 'last_games' games) for each team in the given dataframe.

        :param dataframe: The original dataframe where the 'home_recent_form_points' and 'away_recent_form_points'
        columns will be updated.
        :param last_games: The number of last games to consider for calculating recent form points.
        :param data_column_suffix: The suffix to append to the data column name for home and away teams.
        :param home_teams_group: The DataFrameGroupBy object containing the grouped data by home teams.
        :param away_teams_group: The DataFrameGroupBy object containing the grouped data by away teams.
        :param form_type: The type of recent form points to calculate ('scored' or 'conceded').
        :param swap_home_away: If True, swap home and away teams for calculations.
        :return: None
        """
        # Initialize the recent form columns for home and away teams
        home_update_column: str = (
            f"home_team_{data_column_suffix}_points_{form_type}_last_{last_games}"
        )
        away_update_column: str = (
            f"away_team_{data_column_suffix}_points_{form_type}_last_{last_games}"
        )

        # Calculate the home and away game points using the existing _calculate_points method
        home_data_column: str = f"home_team_{data_column_suffix}_game_points"
        away_data_column: str = f"away_team_{data_column_suffix}_game_points"

        if swap_home_away:
            home_data_column, away_data_column = away_data_column, home_data_column

        # Calculate and update recent form for home teams
        self._calculate_recent_form(
            teams_group=home_teams_group,
            dataframe=dataframe,
            data_column=home_data_column,
            update_column=home_update_column,
            last_games=last_games,
        )

        # Calculate and update recent form for away teams
        self._calculate_recent_form(
            teams_group=away_teams_group,
            dataframe=dataframe,
            data_column=away_data_column,
            update_column=away_update_column,
            last_games=last_games,
        )

    def _calculate_head_to_head_score(
        self,
        dataframe: pd.DataFrame,
        groupby_columns: List[str],
        last_games: int,
        column_suffix: str,
    ) -> None:
        """
        Calculate the Head-to-Head Score for each team based on the last 'last_games' match results between two
        teams.

        :param dataframe: The original dataframe containing match results.
        :param groupby_columns: A list of two columns to use for grouping the dataframe (e.g., ['home_team',
        'away_team']).
        :param last_games: The number of last games to consider for Head-to-Head Performance.
        :param column_suffix: The suffix for the column names indicating the type of match results ('full_time' or
        'half_time').
        :return: None
        """
        # Group the dataframe by the specified columns (home_team and away_team)
        teams_group: DataFrameGroupBy = dataframe.groupby(groupby_columns)

        # Calculate and update the Head-to-Head Performance for each team
        self._calculate_recent_form(
            teams_group=teams_group,
            dataframe=dataframe,
            data_column=f"home_team_{column_suffix}_game_points",
            update_column=f"home_team_{column_suffix}_head_to_head_score_last_{last_games}",
            last_games=last_games,
        )

        self._calculate_recent_form(
            teams_group=teams_group,
            dataframe=dataframe,
            data_column=f"away_team_{column_suffix}_game_points",
            update_column=f"away_team_{column_suffix}_head_to_head_score_last_{last_games}",
            last_games=last_games,
        )

    def _calculate_and_update_recent_form(
        self,
        dataframe: pd.DataFrame,
        last_games: int,
        data_column: str,
        data_column_suffix: str,
        home_teams_group: DataFrameGroupBy,
        away_teams_group: DataFrameGroupBy,
        swap_home_away: bool,
    ) -> None:
        """
        Calculate and update the recent form for the given data column (e.g., goals, shots, shots on target) for each
        team in the given dataframe.

        :param dataframe: The original dataframe where the 'home_recent_form' and 'away_recent_form' columns will be
        updated.
        :param last_games: The number of last games to consider for calculating recent form.
        :param data_column: The name of the data column for which the recent form needs to be calculated (e.g.,
        'goals', 'shots').
        :param data_column_suffix: The suffix to append to the update column names.
        :param home_teams_group: The DataFrameGroupBy object containing the grouped data by home teams.
        :param away_teams_group: The DataFrameGroupBy object containing the grouped data by away teams.
        :param swap_home_away: If True, the data columns for home and away teams will be swapped before calculating and
        updating the recent form. If False, the original data columns for home and away teams will be used.
        :return: None
        """
        # Initialize the recent form columns for home and away teams with suffix
        home_data_column: str = f"home_team_{data_column}"
        away_data_column: str = f"away_team_{data_column}"

        if swap_home_away:
            home_data_column, away_data_column = away_data_column, home_data_column

        home_update_column: str = (
            f"home_team_{data_column}_{data_column_suffix}_last_{last_games}"
        )
        away_update_column: str = (
            f"away_team_{data_column}_{data_column_suffix}_last_{last_games}"
        )

        # Calculate and update recent form for home teams
        self._calculate_recent_form(
            teams_group=home_teams_group,
            dataframe=dataframe,
            data_column=home_data_column,
            update_column=home_update_column,
            last_games=last_games,
        )

        # Calculate and update recent form for away teams
        self._calculate_recent_form(
            teams_group=away_teams_group,
            dataframe=dataframe,
            data_column=away_data_column,
            update_column=away_update_column,
            last_games=last_games,
        )

    @staticmethod
    def _add_season_points_columns(
        dataframe: pd.DataFrame,
        home_teams_group: DataFrameGroupBy,
        away_teams_group: DataFrameGroupBy,
        data_column_suffix: str,
    ) -> None:
        """
        Add season points columns for home and away teams in each group.

        :param dataframe: The original DataFrame.
        :param home_teams_group: The Pandas GroupBy object representing the home teams.
        :param away_teams_group: The Pandas GroupBy object representing the away teams.
        :param data_column_suffix: The suffix to append to the data column name for home and away teams. Should
        represent full time or half time.
        :return: None
        """
        # Define the relevant column names for home_teams_group DataFrame
        selected_columns_home: List[str] = [
            "season_end_year",
            "date",
            "home_team",
            "away_team",
            f"home_team_{data_column_suffix}_game_points",
            f"away_team_{data_column_suffix}_game_points",
        ]

        # Extract DataFrames from home_teams_group containing relevant columns
        home_teams_data: List[pd.DataFrame] = [
            group_data for _, group_data in home_teams_group[selected_columns_home]
        ]

        # Selecting relevant columns from the 'away_teams_group' DataFrame
        selected_columns: List[str] = [
            "season_end_year",
            "date",
            "home_team",
            "away_team",
            f"home_team_{data_column_suffix}_game_points",
            f"away_team_{data_column_suffix}_game_points",
        ]

        # Renaming and swapping column names for away_teams_data
        away_teams_data: List[pd.DataFrame] = [
            group_data.rename(
                columns={
                    "home_team": "away_team",
                    "away_team": "home_team",
                    f"home_team_{data_column_suffix}_game_points": f"away_team_{data_column_suffix}_game_points",
                    f"away_team_{data_column_suffix}_game_points": f"home_team_{data_column_suffix}_game_points",
                }
            )
            for _, group_data in away_teams_group[selected_columns]
        ]

        # Merge home_teams_data and away_teams_data into a single DataFrame.
        all_teams_data: pd.DataFrame = pd.concat(home_teams_data + away_teams_data)

        # Sort the resulting DataFrame by 'home_team' and 'date'.
        all_teams_data.sort_values(by=["home_team", "date"], inplace=True)

        # Group the merged data by 'home_team'.
        all_team_group: DataFrameGroupBy = all_teams_data.groupby(
            "home_team", as_index=False
        )

        # Calculate the season points for each team in the group.
        for team_name, group_data in all_team_group:
            # Extract relevant data for the current team group.
            season_end_year: str = group_data.loc[
                group_data.index[0], "season_end_year"
            ]
            season_points: int = 0

            for index, match in group_data.iterrows():
                # Check if season_end_year changed for the current team.
                if match["season_end_year"] != season_end_year:
                    # For a new season or the first match of the season, reset the season points.
                    season_end_year = match["season_end_year"]
                    season_points = 0

                # Add the points from the current match to the season points.
                season_points += match[f"home_team_{data_column_suffix}_game_points"]

                # Update the season points for the current match in the DataFrame.
                if dataframe.loc[index, "home_team"] == team_name:
                    dataframe.loc[
                        index, f"home_team_{data_column_suffix}_season_points"
                    ] = season_points
                else:
                    dataframe.loc[
                        index, f"away_team_{data_column_suffix}_season_points"
                    ] = season_points

    @staticmethod
    def _calculate_time_since_previous_match(
        dataframe: pd.DataFrame,
        home_teams_group: DataFrameGroupBy,
        away_teams_group: DataFrameGroupBy,
    ) -> None:
        """
        Calculate the time since the previous match for each team in the given DataFrame.

        :param dataframe: The DataFrame containing the football match data.
        :param home_teams_group: The Pandas GroupBy object representing the home teams.
        :param away_teams_group: The Pandas GroupBy object representing the away teams.
        :return: None. The original DataFrame will be updated with additional columns for time since the previous match
        for each home and away team.
        """
        # Extract DataFrames from home_teams_group and away_teams_group containing relevant columns.
        home_teams_data: List[pd.DataFrame] = [
            group_data
            for _, group_data in home_teams_group[
                ["season_end_year", "date", "home_team", "away_team"]
            ]
        ]
        away_teams_data: List[pd.DataFrame] = [
            group_data.rename(
                columns={"home_team": "away_team", "away_team": "home_team"}
            )
            for _, group_data in away_teams_group[
                ["season_end_year", "date", "home_team", "away_team"]
            ]
        ]

        # Merge home_teams_data and away_teams_data into a single DataFrame.
        all_teams_data: pd.DataFrame = pd.concat(home_teams_data + away_teams_data)

        # Sort the resulting DataFrame by 'home_team' and 'date'.
        all_teams_data.sort_values(by=["home_team", "date"], inplace=True)

        # Group the merged data by 'home_team'.
        all_team_group: DataFrameGroupBy = all_teams_data.groupby(
            "home_team", as_index=False
        )

        # Calculate the time since the previous match for each team in the group.
        for team_name, group_data in all_team_group:
            # Extract relevant data for the current team group.
            season_end_year: str = group_data.loc[group_data.index[0]][
                "season_end_year"
            ]
            last_match_date: Optional[pd.Timestamp] = None

            for index, match in group_data.iterrows():
                # Check if season_end_year changed for the current team.
                if match["season_end_year"] != season_end_year:
                    # For a new season or the first match of the season, set the last match date to None.
                    season_end_year = match["season_end_year"]
                    last_match_date = None

                if last_match_date is None:
                    # For the first match of the season, set NaN as there is no previous match.
                    time_since_previous_match: np.nan = np.nan
                else:
                    # Calculate the time since the previous match within the same season.
                    time_since_previous_match: int = (
                        match["date"] - last_match_date
                    ).days

                # Update the last match date for the current team and season.
                last_match_date = match["date"]

                # Add the calculated time since previous match as a new column to the dataframe.
                if dataframe.loc[index]["home_team"] == team_name:
                    dataframe.loc[index, "home_team_time_since_previous_match"] = (
                        time_since_previous_match
                    )
                else:
                    dataframe.loc[index, "away_team_time_since_previous_match"] = (
                        time_since_previous_match
                    )

    @staticmethod
    def _calculate_column_difference(
        dataframe: pd.DataFrame,
        statistics_dataframe: pd.DataFrame,
        column1: str,
        column2: str,
        new_column_name: str,
    ) -> None:
        """
        Calculate the difference between two columns in the DataFrame and add it as a new column in the statistics
        DataFrame.

        :param dataframe: Input DataFrame containing match data.
        :param statistics_dataframe: DataFrame to store statistics.
        :param column1: The name of the first column.
        :param column2: The name of the second column.
        :param new_column_name: The name of the new column to be added with the difference.
        :return: None (The new column will be added to the statistics DataFrame).
        """
        # Calculate the difference and set the value directly using .loc
        statistics_dataframe.loc[:, new_column_name] = (
            dataframe.loc[:, column1] - dataframe.loc[:, column2]
        )

    @staticmethod
    def _update_cumulative_sum_generic(
        dataframe: pd.DataFrame, columns_mapping: List[Tuple[str, str, bool]]
    ) -> None:
        """
        Calculate the cumulative sum for each team in each match based on the provided data columns.

        :param dataframe: The input DataFrame containing match data.
        :param columns_mapping: A list of tuples, where each tuple contains the data column, its corresponding update
                                column, and a boolean indicating whether to swap the data column of home and away
                                teams or not.
        :return: None
        """
        # Creating dictionaries to store the cumulative sums for each team and data column
        cumulative_sums: Dict[str, Dict[str, List[float]]] = {
            f"swapped_{data_column}" if swap_home_away else data_column: {}
            for data_column, _, swap_home_away in columns_mapping
        }

        season_end_year: int = 1993

        for index, row in dataframe.iterrows():
            if row["season_end_year"] != season_end_year:
                cumulative_sums = {
                    f"swapped_{data_column}" if swap_home_away else data_column: {}
                    for data_column, _, swap_home_away in columns_mapping
                }
                season_end_year = row["season_end_year"]

            home_team: str = row["home_team"]
            away_team: str = row["away_team"]

            for data_column, update_column, swap_home_away in columns_mapping:
                home_cumulative_key: str = f"home_team_{data_column}"
                away_cumulative_key: str = f"away_team_{data_column}"

                if swap_home_away:
                    data_column: str = f"swapped_{data_column}"
                    home_cumulative_key, away_cumulative_key = (
                        away_cumulative_key,
                        home_cumulative_key,
                    )

                home_team_key: str = f"home_team_{home_team}"
                away_team_key: str = f"away_team_{away_team}"

                # Update the cumulative sum for home_team for the current data column
                if home_team_key not in cumulative_sums[data_column]:
                    cumulative_sums[data_column][home_team_key] = [0, 0]
                cumulative_sums[data_column][home_team_key][0] += row[
                    home_cumulative_key
                ]
                cumulative_sums[data_column][home_team_key][1] += 1

                # Update the cumulative sum for away_team for the current data column
                if away_team_key not in cumulative_sums[data_column]:
                    cumulative_sums[data_column][away_team_key] = [0, 0]
                cumulative_sums[data_column][away_team_key][0] += row[
                    away_cumulative_key
                ]
                cumulative_sums[data_column][away_team_key][1] += 1

                # Update the DataFrame with the cumulative sums for each match and data column
                home_update_column: str = f"home_team_{update_column}"
                away_update_column: str = f"away_team_{update_column}"

                dataframe.at[index, home_update_column] = (
                    cumulative_sums[data_column][home_team_key][0]
                    / cumulative_sums[data_column][home_team_key][1]
                )
                dataframe.at[index, away_update_column] = (
                    cumulative_sums[data_column][away_team_key][0]
                    / cumulative_sums[data_column][away_team_key][1]
                )

    def _feature_engineering(self, dataframe: pd.DataFrame) -> None:
        """
        Preprocesses the dataset by applying feature engineering.

        :param dataframe: Input dataset as a pandas DataFrame.

        :return: None
        """
        # Convert the 'season_end_year' column from string to integer data type
        dataframe["season_end_year"] = dataframe["season_end_year"].astype(int)

        # Convert the 'date' column to timestamps if it is in string format
        dataframe["date"] = pd.to_datetime(dataframe["date"], dayfirst=True).dt.date

        # Extract season from the 'game_date' column
        dataframe["season"] = dataframe["date"].apply(self._get_season)

        # Extract season from the 'game_date' column
        self._add_matchweek_column(dataframe=dataframe)

        # Calculate and assign team points for 'full_time' and 'half_time' data columns
        self._calculate_and_assign_team_points(dataframe=dataframe)

        # Group the DataFrame by each team (home and away)
        home_teams_group: DataFrameGroupBy = dataframe.groupby(
            "home_team", as_index=True
        )
        away_teams_group: DataFrameGroupBy = dataframe.groupby(
            "away_team", as_index=True
        )

        for last_games in MatchOutcomePredictor._LAST_GAMES_LIST:
            # Calculate and update recent form full-time points scored
            self._calculate_and_update_recent_form_points(
                dataframe=dataframe,
                last_games=last_games,
                data_column_suffix="full_time",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                form_type="scored",
                swap_home_away=False,
            )

            # Calculate and update recent form full-time points conceded
            self._calculate_and_update_recent_form_points(
                dataframe=dataframe,
                last_games=last_games,
                data_column_suffix="full_time",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                form_type="conceded",
                swap_home_away=True,
            )

            # Calculate and update recent form half-time points scored
            self._calculate_and_update_recent_form_points(
                dataframe=dataframe,
                last_games=last_games,
                data_column_suffix="half_time",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                form_type="scored",
                swap_home_away=False,
            )

            # Calculate and update recent form half-time points conceded
            self._calculate_and_update_recent_form_points(
                dataframe=dataframe,
                last_games=last_games,
                data_column_suffix="half_time",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                form_type="conceded",
                swap_home_away=True,
            )

            # Calculate and update recent form full-time goals

            # Calculate and update recent form goals scored
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="full_time_goals",
                data_column_suffix="scored",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form goals conceded
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="full_time_goals",
                data_column_suffix="conceded",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form half-time goals

            # Calculate and update recent form goals scored
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="half_time_goals",
                data_column_suffix="scored",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form goals conceded
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="half_time_goals",
                data_column_suffix="conceded",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form shots

            # Calculate and update recent form for total shots taken
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="shots",
                data_column_suffix="taken",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total shots faced
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="shots",
                data_column_suffix="faced",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form shots on target

            # Calculate and update recent form for total shots on target taken by home teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="shots_on_target",
                data_column_suffix="taken",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total shots on target faced for away teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="shots_on_target",
                data_column_suffix="faced",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form corners

            # Calculate and update recent form for total corners won by home teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="corners",
                data_column_suffix="won",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total corners conceded by away teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="corners",
                data_column_suffix="conceded",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form fouls

            # Calculate and update recent form for total fouls committed by home teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="fouls",
                data_column_suffix="committed",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total fouls received away teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="fouls",
                data_column_suffix="received",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form yellow cards

            # Calculate and update recent form for total yellow cards received by home teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="yellow_cards",
                data_column_suffix="given",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total yellow cards received against away teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="yellow_cards",
                data_column_suffix="given_to_opponent",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate and update recent form red cards

            # Calculate and update recent form for total red cards received by home teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="red_cards",
                data_column_suffix="given",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=False,
            )

            # Calculate and update recent form for total red cards received against away teams
            self._calculate_and_update_recent_form(
                dataframe=dataframe,
                last_games=last_games,
                data_column="red_cards",
                data_column_suffix="given_to_opponent",
                home_teams_group=home_teams_group,
                away_teams_group=away_teams_group,
                swap_home_away=True,
            )

            # Calculate head-to-head full-time score
            self._calculate_head_to_head_score(
                dataframe=dataframe,
                groupby_columns=["home_team", "away_team"],
                last_games=last_games,
                column_suffix="full_time",
            )

            # Calculate head-to-head half-time score
            self._calculate_head_to_head_score(
                dataframe=dataframe,
                groupby_columns=["home_team", "away_team"],
                last_games=last_games,
                column_suffix="half_time",
            )

        # Add full-time season points columns
        self._add_season_points_columns(
            dataframe=dataframe,
            home_teams_group=home_teams_group,
            away_teams_group=away_teams_group,
            data_column_suffix="full_time",
        )

        # Add half-time season points columns
        self._add_season_points_columns(
            dataframe=dataframe,
            home_teams_group=home_teams_group,
            away_teams_group=away_teams_group,
            data_column_suffix="half_time",
        )

        # Calculate time since previous match
        self._calculate_time_since_previous_match(
            dataframe=dataframe,
            home_teams_group=home_teams_group,
            away_teams_group=away_teams_group,
        )

        self._update_cumulative_sum_generic(
            dataframe=dataframe,
            columns_mapping=[
                ("full_time_game_points", "cumulative_full_time_points_scored", False),
                ("full_time_game_points", "cumulative_full_time_points_conceded", True),
                ("half_time_game_points", "cumulative_half_time_points_scored", False),
                ("half_time_game_points", "cumulative_half_time_points_conceded", True),
                ("full_time_goals", "cumulative_full_time_goals_scored", False),
                ("full_time_goals", "cumulative_full_time_goals_conceded", True),
                ("half_time_goals", "cumulative_half_time_goals_scored", False),
                ("half_time_goals", "cumulative_half_time_goals_conceded", True),
                ("shots", "cumulative_shots_taken", False),
                ("shots", "cumulative_shots_faced", True),
                ("shots_on_target", "cumulative_shots_on_target_taken", False),
                ("shots_on_target", "cumulative_shots_on_target_faced", True),
                ("corners", "cumulative_corners_won", False),
                ("corners", "cumulative_corners_conceded", True),
                ("fouls", "cumulative_fouls_committed", False),
                ("fouls", "cumulative_fouls_received", True),
                ("yellow_cards", "cumulative_yellow_cards_given", False),
                ("yellow_cards", "cumulative_yellow_cards_given_to_opponent", True),
                ("red_cards", "cumulative_red_cards_given", False),
                ("red_cards", "cumulative_red_cards_given_to_opponent", True),
            ],
        )

    def _calculate_all_feature_differences(
        self, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the differences between feature engineering statistics and update them in the statistics DataFrame.

        :param dataframe: Input dataset as a pandas DataFrame.

        :return: DataFrame containing the statistics with the differences and other features.
        """
        # Create a new DataFrame 'statistics_dataframe'
        statistics_dataframe: pd.DataFrame = pd.DataFrame()

        # Set the prefix for column names
        home_team_prefix: str = "home_team"
        away_team_prefix: str = "away_team"

        # Iterate over the list of last_games
        for last_games in MatchOutcomePredictor._LAST_GAMES_LIST:
            # Set the suffix for column names
            games_suffix: str = f"last_{last_games}"

            # Calculate and update the differences for full-time points scored
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_full_time_points_scored_{games_suffix}",
                column2=f"{away_team_prefix}_full_time_points_scored_{games_suffix}",
                new_column_name=f"difference_full_time_points_scored_{games_suffix}",
            )

            # Calculate and update the differences for full-time points conceded
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_full_time_points_conceded_{games_suffix}",
                column2=f"{away_team_prefix}_full_time_points_conceded_{games_suffix}",
                new_column_name=f"difference_full_time_points_conceded_{games_suffix}",
            )

            # Calculate and update the differences for half-time points scored
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_half_time_points_scored_{games_suffix}",
                column2=f"{away_team_prefix}_half_time_points_scored_{games_suffix}",
                new_column_name=f"difference_half_time_points_scored_{games_suffix}",
            )

            # Calculate and update the differences for half-time points conceded
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_half_time_points_conceded_{games_suffix}",
                column2=f"{away_team_prefix}_half_time_points_conceded_{games_suffix}",
                new_column_name=f"difference_half_time_points_conceded_{games_suffix}",
            )

            # Calculate and update the differences for full-time goals scored
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_full_time_goals_scored_{games_suffix}",
                column2=f"{away_team_prefix}_full_time_goals_scored_{games_suffix}",
                new_column_name=f"difference_full_time_goals_scored_{games_suffix}",
            )

            # Calculate and update the differences for half-time goals scored
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_half_time_goals_scored_{games_suffix}",
                column2=f"{away_team_prefix}_half_time_goals_scored_{games_suffix}",
                new_column_name=f"difference_half_time_goals_scored_{games_suffix}",
            )

            # Calculate and update the differences for full-time goals conceded
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_full_time_goals_conceded_{games_suffix}",
                column2=f"{away_team_prefix}_full_time_goals_conceded_{games_suffix}",
                new_column_name=f"difference_full_time_goals_conceded_{games_suffix}",
            )

            # Calculate and update the differences for half-time goals conceded
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_half_time_goals_conceded_{games_suffix}",
                column2=f"{away_team_prefix}_half_time_goals_conceded_{games_suffix}",
                new_column_name=f"difference_half_time_goals_conceded_{games_suffix}",
            )

            # Calculate and update the differences for shots taken
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_shots_taken_{games_suffix}",
                column2=f"{away_team_prefix}_shots_taken_{games_suffix}",
                new_column_name=f"difference_shots_taken_{games_suffix}",
            )

            # Calculate and update the differences for shots faced
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_shots_faced_{games_suffix}",
                column2=f"{away_team_prefix}_shots_faced_{games_suffix}",
                new_column_name=f"difference_shots_faced_{games_suffix}",
            )

            # Calculate and update the differences for shots on target taken
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_shots_on_target_taken_{games_suffix}",
                column2=f"{away_team_prefix}_shots_on_target_taken_{games_suffix}",
                new_column_name=f"difference_shots_on_target_taken_{games_suffix}",
            )

            # Calculate and update the differences for shots on target faced
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_shots_on_target_faced_{games_suffix}",
                column2=f"{away_team_prefix}_shots_on_target_faced_{games_suffix}",
                new_column_name=f"difference_shots_on_target_faced_{games_suffix}",
            )

            # Calculate and update the differences for corners won
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_corners_won_{games_suffix}",
                column2=f"{away_team_prefix}_corners_won_{games_suffix}",
                new_column_name=f"difference_corners_won_{games_suffix}",
            )

            # Calculate and update the differences for corners conceded
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_corners_conceded_{games_suffix}",
                column2=f"{away_team_prefix}_corners_conceded_{games_suffix}",
                new_column_name=f"difference_corners_conceded_{games_suffix}",
            )

            # Calculate and update the differences for fouls committed
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_fouls_committed_{games_suffix}",
                column2=f"{away_team_prefix}_fouls_committed_{games_suffix}",
                new_column_name=f"difference_fouls_committed_{games_suffix}",
            )

            # Calculate and update the differences for fouls received
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_fouls_received_{games_suffix}",
                column2=f"{away_team_prefix}_fouls_received_{games_suffix}",
                new_column_name=f"difference_fouls_received_{games_suffix}",
            )

            # Calculate and update the differences for yellow cards given
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_yellow_cards_given_{games_suffix}",
                column2=f"{away_team_prefix}_yellow_cards_given_{games_suffix}",
                new_column_name=f"difference_yellow_cards_given_{games_suffix}",
            )

            # Calculate and update the differences for yellow cards given_to_opponent
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_yellow_cards_given_to_opponent_{games_suffix}",
                column2=f"{away_team_prefix}_yellow_cards_given_to_opponent_{games_suffix}",
                new_column_name=f"difference_yellow_cards_given_to_opponent_{games_suffix}",
            )

            # Calculate and update the differences for red cards given
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_red_cards_given_{games_suffix}",
                column2=f"{away_team_prefix}_red_cards_given_{games_suffix}",
                new_column_name=f"difference_red_cards_given_{games_suffix}",
            )

            # Calculate and update the differences for red cards given_to_opponent
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_red_cards_given_to_opponent_{games_suffix}",
                column2=f"{away_team_prefix}_red_cards_given_to_opponent_{games_suffix}",
                new_column_name=f"difference_red_cards_given_to_opponent_{games_suffix}",
            )

            # Calculate and update the differences for full-time head-to-head score
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_full_time_head_to_head_score_{games_suffix}",
                column2=f"{away_team_prefix}_full_time_head_to_head_score_{games_suffix}",
                new_column_name=f"difference_full_time_head_to_head_score_{games_suffix}",
            )

            # Calculate and update the differences for half-time head-to-head score
            self._calculate_column_difference(
                dataframe=dataframe,
                statistics_dataframe=statistics_dataframe,
                column1=f"{home_team_prefix}_half_time_head_to_head_score_{games_suffix}",
                column2=f"{away_team_prefix}_half_time_head_to_head_score_{games_suffix}",
                new_column_name=f"difference_half_time_head_to_head_score_{games_suffix}",
            )

        # Calculate and update the differences for full-time season points
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_full_time_season_points",
            column2=f"{away_team_prefix}_full_time_season_points",
            new_column_name="difference_full_time_season_points",
        )

        # Calculate and update the differences for half-time season points
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_half_time_season_points",
            column2=f"{away_team_prefix}_half_time_season_points",
            new_column_name="difference_half_time_season_points",
        )

        # Calculate and update the differences for time since previous match
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_time_since_previous_match",
            column2=f"{away_team_prefix}_time_since_previous_match",
            new_column_name=f"difference_time_since_previous_match",
        )

        # Calculate and update the differences for cumulative full-time points scored
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_full_time_points_scored",
            column2=f"{away_team_prefix}_cumulative_full_time_points_scored",
            new_column_name="difference_cumulative_full_time_points_scored",
        )

        # Calculate and update the differences for cumulative full-time points conceded
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_full_time_points_conceded",
            column2=f"{away_team_prefix}_cumulative_full_time_points_conceded",
            new_column_name="difference_cumulative_full_time_points_conceded",
        )

        # Calculate and update the differences for cumulative half-time points scored
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_half_time_points_scored",
            column2=f"{away_team_prefix}_cumulative_half_time_points_scored",
            new_column_name="difference_cumulative_half_time_points_scored",
        )

        # Calculate and update the differences for cumulative half-time points conceded
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_half_time_points_conceded",
            column2=f"{away_team_prefix}_cumulative_half_time_points_conceded",
            new_column_name="difference_cumulative_half_time_points_conceded",
        )

        # Calculate and update the differences for cumulative full-time goals scored
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_full_time_goals_scored",
            column2=f"{away_team_prefix}_cumulative_full_time_goals_scored",
            new_column_name="difference_cumulative_full_time_goals_scored",
        )

        # Calculate and update the differences for cumulative full-time goals conceded
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_full_time_goals_conceded",
            column2=f"{away_team_prefix}_cumulative_full_time_goals_conceded",
            new_column_name="difference_cumulative_full_time_goals_conceded",
        )

        # Calculate and update the differences for cumulative half-time goals scored
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_half_time_goals_scored",
            column2=f"{away_team_prefix}_cumulative_half_time_goals_scored",
            new_column_name="difference_cumulative_half_time_goals_scored",
        )

        # Calculate and update the differences for cumulative half-time goals conceded
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_half_time_goals_conceded",
            column2=f"{away_team_prefix}_cumulative_half_time_goals_conceded",
            new_column_name="difference_cumulative_half_time_goals_conceded",
        )

        # Calculate and update the differences for cumulative shots taken
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_shots_taken",
            column2=f"{away_team_prefix}_cumulative_shots_taken",
            new_column_name="difference_cumulative_shots_taken",
        )

        # Calculate and update the differences for cumulative shots faced
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_shots_faced",
            column2=f"{away_team_prefix}_cumulative_shots_faced",
            new_column_name="difference_cumulative_shots_faced",
        )

        # Calculate and update the differences for cumulative shots on target taken
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_shots_on_target_taken",
            column2=f"{away_team_prefix}_cumulative_shots_on_target_taken",
            new_column_name="difference_cumulative_shots_on_target_taken",
        )

        # Calculate and update the differences for cumulative shots on target faced
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_shots_on_target_faced",
            column2=f"{away_team_prefix}_cumulative_shots_on_target_faced",
            new_column_name="difference_cumulative_shots_on_target_faced",
        )

        # Calculate and update the differences for cumulative corners won
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_corners_won",
            column2=f"{away_team_prefix}_cumulative_corners_won",
            new_column_name="difference_cumulative_corners_won",
        )

        # Calculate and update the differences for cumulative corners conceded
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_corners_conceded",
            column2=f"{away_team_prefix}_cumulative_corners_conceded",
            new_column_name="difference_cumulative_corners_conceded",
        )

        # Calculate and update the differences for cumulative fouls committed
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_fouls_committed",
            column2=f"{away_team_prefix}_cumulative_fouls_committed",
            new_column_name="difference_cumulative_fouls_committed",
        )

        # Calculate and update the differences for cumulative fouls received
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_fouls_received",
            column2=f"{away_team_prefix}_cumulative_fouls_received",
            new_column_name="difference_cumulative_fouls_received",
        )

        # Calculate and update the differences for cumulative yellow cards given
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_yellow_cards_given",
            column2=f"{away_team_prefix}_cumulative_yellow_cards_given",
            new_column_name="difference_cumulative_yellow_cards_given",
        )

        # Calculate and update the differences for cumulative yellow cards given to opponent
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_yellow_cards_given_to_opponent",
            column2=f"{away_team_prefix}_cumulative_yellow_cards_given_to_opponent",
            new_column_name="difference_cumulative_yellow_cards_given_to_opponent",
        )

        # Calculate and update the differences for cumulative red cards given
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_red_cards_given",
            column2=f"{away_team_prefix}_cumulative_red_cards_given",
            new_column_name="difference_cumulative_red_cards_given",
        )

        # Calculate and update the differences for cumulative red cards given to opponent
        self._calculate_column_difference(
            dataframe=dataframe,
            statistics_dataframe=statistics_dataframe,
            column1=f"{home_team_prefix}_cumulative_red_cards_given_to_opponent",
            column2=f"{away_team_prefix}_cumulative_red_cards_given_to_opponent",
            new_column_name="difference_cumulative_red_cards_given_to_opponent",
        )

        return statistics_dataframe

    @staticmethod
    def _select_and_concat_columns(
        statistics_dataframe: pd.DataFrame, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Selects specific columns from a DataFrame and concatenates them to an existing DataFrame.

        :param statistics_dataframe: DataFrame to which selected columns will be concatenated.
        :param dataframe: DataFrame containing the columns to be selected and concatenated.

        :return: A new DataFrame containing the selected columns concatenated to the existing DataFrame.
        """
        # List of columns to be selected and concatenated
        selected_columns: List[str] = [
            "season_end_year",
            "season",
            "matchweek",
            "full_time_result",
        ]

        # Select columns from the 'dataframe' and concatenate to 'statistics_dataframe'
        concatenated_dataframe: pd.DataFrame = pd.concat(
            [statistics_dataframe, dataframe[selected_columns]], axis=1
        )

        return concatenated_dataframe

    @staticmethod
    def _preprocess_data(
        statistics_dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses the dataset by applying feature engineering and encoding.

        :param statistics_dataframe: Input dataset as a pandas DataFrame.

        :return: Tuple containing preprocessed features (X) and target labels (y).
        """
        # Map the 'full_time_result' column to numerical values (Home Win: 2, Draw: 1, Away Win: 0)
        result_mapping: Dict[str, int] = {"H": 2, "D": 1, "A": 0}
        statistics_dataframe["full_time_result"]: pd.Series = statistics_dataframe[
            "full_time_result"
        ].map(result_mapping)

        # Select features (input) and target (output) variables
        all_columns: List[str] = statistics_dataframe.columns.tolist()
        all_columns.remove("full_time_result")

        features: List[str] = all_columns
        target: str = "full_time_result"

        X: pd.DataFrame = statistics_dataframe[features]
        y: pd.Series = statistics_dataframe[target]

        return X, y

    @staticmethod
    def _tune_model(
        model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]:
        """
        Performs hyperparameter tuning on the given model using RandomizedSearchCV.

        :param model: Model to be tuned, chosen from XGBClassifier, CatBoostClassifier, LGBMClassifier.
        :param X_train: Training features as a pandas DataFrame.
        :param y_train: Training labels as a pandas Series.

        :return: Tuned model of the same type as the input model.
        """
        # Define the hyperparameter search space based on the type of model
        param_dist: Dict[str, List[Any]] = {}

        if isinstance(model, XGBClassifier):
            param_dist = {
                "objective": ["multi:softmax"],
                "num_class": [3],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [100, 200, 300, 400, 500],
                "gamma": [0, 0.1, 0.2, 0.3, 0.4],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1, 10],
                "reg_lambda": [0, 0.01, 0.1, 1, 10],
            }
        elif isinstance(model, CatBoostClassifier):
            param_dist = {
                "depth": [3, 5, 7, 9],
                "learning_rate": [0.1, 0.01, 0.001],
                "iterations": [100, 200, 300, 400, 500],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
                "border_count": [32, 64, 128],
                "class_weights": [[1, 1, 1], [1, 5, 10], [1, 10, 5]],
            }
        elif isinstance(model, LGBMClassifier):
            param_dist = {
                "objective": ["multiclass"],
                "num_class": [3],
                "metric": ["multi_logloss"],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [100, 200, 300, 400, 500],
                "num_leaves": [31, 63, 127, 255],
                "min_child_samples": [1, 5, 10, 20, 30],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1, 10],
                "reg_lambda": [0, 0.01, 0.1, 1, 10],
            }

        # Initialize the RandomizedSearchCV with 5-fold cross-validation
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring="f1_macro",
            cv=5,
            n_iter=50,
            n_jobs=-1,
            random_state=42,
        )

        # Perform the random search to find the best hyperparameters using the training data
        random_search.fit(X_train, y_train)

        # Get the best estimator (model) from the random search
        best_model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier] = (
            random_search.best_estimator_
        )

        # Return the best tuned model
        return best_model

    def _select_best_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]:
        """
        Selects the best model based on evaluation metrics for multiclass classification.

        :param X: Features as a pandas DataFrame.
        :param y: Labels as a pandas Series.

        :return: The best trained model.
        """
        # Split the dataset into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Define the scoring function based on the type of classification problem
        scoring_function: Callable[[Any, Any, Any], float] = (
            lambda estimator, X, y: balanced_accuracy_score(y, estimator.predict(X))
        )

        models: List[Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]] = [
            XGBClassifier(),
            CatBoostClassifier(loss_function="MultiClass"),
            LGBMClassifier(),
        ]

        best_model: Optional[
            Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]
        ] = None
        best_score: float = 0

        for model in models:
            # Call the '_tune_model' method to find the best hyperparameters for the current model
            tuned_model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier] = (
                self._tune_model(model, X_train, y_train)
            )

            # Evaluate the tuned model on the test set
            score: float = scoring_function(tuned_model, X_test, y_test)

            # Check if the current model's score is better than the previous best score
            if best_score < score:
                best_score = score
                best_model = tuned_model

        return best_model

    @staticmethod
    def _save_processed_data_and_model(
        save_path: str,
        dataframe: pd.DataFrame,
        model: Optional[Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]],
    ) -> None:
        """
        Save the processed DataFrame and trained model.

        :param save_path: The path to save the processed DataFrame and model.
        :param dataframe: The processed DataFrame to be saved.
        :param model: The trained model to be saved.
        :return: None

        :raises IOError: If an error occurs while saving the files.
        """
        try:
            # Ensure the save directory exists or create it
            os.makedirs(save_path, exist_ok=True)

            # Define file paths
            processed_data_path: str = os.path.join(
                save_path, "processed_dataframe.csv"
            )
            model_path: str = os.path.join(save_path, "trained_model.pkl")

            # Write the DataFrame to a CSV file using 'NaN' as the representation for missing values
            dataframe.to_csv(processed_data_path, index=False, na_rep="NaN")

            # Save the trained model using joblib
            joblib.dump(model, model_path)
        except Exception as e:
            raise IOError(
                f"An error occurred while saving the processed DataFrame and model: {e}"
            )

    def create_match_outcome_predictor(self) -> None:
        """
        Creates a match outcome predictor and saves the processed DataFrame and model.

        :return: None

        :raises ValueError: If data_path or save_path is not a non-empty string or if they're None.
        :raises IOError: If an error occurs while loading the data or saving the files.
        """
        input_data_path: str = MatchOutcomePredictor._INPUT_DATA_PATH
        saved_model_directory: str = MatchOutcomePredictor._SAVED_MODEL_DIRECTORY

        # Check if data_path and save_path are strings and are non-empty
        if not (isinstance(input_data_path, str) and input_data_path) or not (
            isinstance(saved_model_directory, str) and saved_model_directory
        ):
            raise ValueError(
                "Both data_path and save_path must be non-empty strings and must be provided."
            )

        try:
            # Load data from the given path
            dataframe: pd.DataFrame = pd.read_csv(filepath_or_buffer=input_data_path)
        except IOError as e:
            raise IOError(f"An error occurred while loading the data: {e}")

        # Validate the loaded DataFrame
        self._validate_dataframe(dataframe=dataframe)

        # Perform feature engineering and calculations
        self._feature_engineering(dataframe=dataframe)

        # Calculate feature differences
        statistics_dataframe: pd.DataFrame = self._calculate_all_feature_differences(
            dataframe=dataframe
        )

        # Select and concatenate relevant columns
        statistics_dataframe = self._select_and_concat_columns(
            dataframe=dataframe, statistics_dataframe=statistics_dataframe
        )

        # Preprocess data to prepare it for modeling
        X, y = self._preprocess_data(statistics_dataframe=statistics_dataframe)

        # Select the best model using the preprocessed data
        selected_model: Optional[
            Union[XGBClassifier, CatBoostClassifier, LGBMClassifier]
        ] = self._select_best_model(X=X, y=y)

        # Save the processed DataFrame and model
        self._save_processed_data_and_model(
            save_path=saved_model_directory, dataframe=dataframe, model=selected_model
        )

    @staticmethod
    def _update_team_statistics_for_prediction(
        dataframe: pd.DataFrame,
        team_name: str,
        team_type: str,
        opponent_team_type: str,
        season_end_year: int,
        last_games: int,
        columns_mapping: List[Tuple[str, str, bool]],
        teams_statistics: pd.DataFrame,
    ) -> None:
        """
        Update team statistics for a specific team in the team_statistics DataFrame based on specified parameters.

        :param dataframe: Input DataFrame containing match data.
        :param team_name: Name of the team to calculate statistics for.
        :param team_type: Type of team (home_team or away_team).
        :param opponent_team_type: Type of opponent team (away_team or home_team).
        :param season_end_year: Ending year of the season.
        :param last_games: Number of last games to consider.
        :param columns_mapping: List of tuples containing (data_column, update_column, swap_data).
        :param teams_statistics: DataFrame to update with calculated statistics.

        :return: None.
        """
        # Filter the data for the specified team and team type, considering the current season and the previous season
        team_data: pd.DataFrame = dataframe[
            (
                (dataframe[team_type] == team_name)
                & (
                    (dataframe["season_end_year"] == season_end_year)
                    | (dataframe["season_end_year"] == season_end_year - 1)
                )
            )
        ]

        # Get the length of the filtered data
        team_data_length = len(team_data)

        if last_games <= team_data_length:
            # Get the last 'last_games' games for the team
            team_last_games: pd.DataFrame = team_data.tail(last_games)

            for data_column, update_column, swap_data in columns_mapping:
                dynamic_data_column: str = (
                    f"{opponent_team_type}_{data_column}"
                    if swap_data
                    else f"{team_type}_{data_column}"
                )

                # Calculate the mean of the dynamic data column for the last games
                mean_value: float = team_last_games[dynamic_data_column].mean()

                # Generate the dynamic update column name with last_games
                dynamic_update_column: str = (
                    f"{team_type}_{update_column}_last_{last_games}"
                )

                # Update the team_statistics DataFrame with the calculated mean value
                teams_statistics[dynamic_update_column] = [mean_value]
        else:
            # If the team has played fewer games than required for analysis, add the missing data from the calculated
            # mean for the three promoted clubs from the previous season.
            promoted_clubs_last_games: pd.DataFrame = dataframe[
                (
                    dataframe[team_type].isin(
                        MatchOutcomePredictor._PROMOTED_CLUBS_2023_24
                    )
                )
                & (dataframe["season_end_year"] == 2024)
            ].tail(
                (last_games - team_data_length)
                * len(MatchOutcomePredictor._PROMOTED_CLUBS_2023_24)
            )

            for data_column, update_column, swap_data in columns_mapping:
                dynamic_data_column: str = (
                    f"{opponent_team_type}_{data_column}"
                    if swap_data
                    else f"{team_type}_{data_column}"
                )

                # Calculate the mean of the dynamic data column for the last games
                mean_value: float = (
                    team_data[dynamic_data_column].sum()
                    + (
                        promoted_clubs_last_games[dynamic_data_column].sum()
                        / len(MatchOutcomePredictor._PROMOTED_CLUBS_2023_24)
                    )
                ) / last_games

                # Generate the dynamic update column name with last_games
                dynamic_update_column: str = (
                    f"{team_type}_{update_column}_last_{last_games}"
                )

                # Update the team_statistics DataFrame with the calculated mean value
                teams_statistics[dynamic_update_column] = [mean_value]

    @staticmethod
    def _update_head_to_head_score_for_prediction(
        dataframe: pd.DataFrame,
        team_name: str,
        opponent_name: str,
        team_type: str,
        opponent_team_type: str,
        season_end_year: int,
        last_games: int,
        teams_statistics: pd.DataFrame,
    ) -> None:
        """
        Update head-to-head performance statistics for a specific team in the team_statistics DataFrame.

        :param dataframe: The original dataframe containing match results.
        :param team_name: Name of the team to update head-to-head performance for.
        :param opponent_name: Name of the opponent team.
        :param team_type: Type of the team ('home_team' or 'away_team').
        :param opponent_team_type: Type of the opponent team ('home_team' or 'away_team').
        :param season_end_year: Ending year of the season.
        :param last_games: The number of last games to consider for Head-to-Head Performance.
        :param teams_statistics: DataFrame to update with calculated statistics.

        :return: None.
        """
        # Calculate the start year based on the season_end_year and the desired last_games value
        start_year: int = season_end_year - last_games

        # Filter the dataframe for matches between the specified team and opponent
        team_group: pd.DataFrame = dataframe[
            (dataframe[team_type] == team_name)
            & (dataframe[opponent_team_type] == opponent_name)
            & (dataframe["season_end_year"] >= start_year)
        ]

        # Get the last 'last_games' matches for the team against the opponent
        teams_last_games: pd.DataFrame = team_group.tail(last_games)

        for column_suffix in ["full_time", "half_time"]:
            # Calculate the head-to-head score by summing the 'team_type' game points
            head_to_head_score: Union[np.nan, pd.DataFrame] = (
                np.nan
                if teams_last_games.empty
                else teams_last_games[f"{team_type}_{column_suffix}_game_points"].sum()
            )

            # Create the name of the column to update in 'team_statistics'
            update_column: str = (
                f"{team_type}_{column_suffix}_head_to_head_score_last_{last_games}"
            )

            # Update the 'team_statistics' DataFrame with the calculated head-to-head score
            teams_statistics[update_column] = [head_to_head_score]

    @staticmethod
    def _calculate_season_points_for_chunk(
        dataframe: pd.DataFrame, team_name: str
    ) -> Tuple[int, int]:
        """
        Calculate season points for a specific team within a chunk of data.

        :param dataframe: DataFrame containing a chunk of consecutive game data.
        :param team_name: Name of the team to calculate season points for.
        :return: Tuple containing the total full-time points and half-time points for the chunk.
        """
        season_full_time_points: int = 0
        season_half_time_points: int = 0

        for index, row in dataframe.iterrows():
            if row["home_team"] == team_name:
                season_full_time_points += row["home_team_full_time_game_points"]
                season_half_time_points += row["home_team_half_time_game_points"]
            else:
                season_full_time_points += row["away_team_full_time_game_points"]
                season_half_time_points += row["away_team_half_time_game_points"]

        return season_full_time_points, season_half_time_points

    def _calculate_season_team_points_for_prediction(
        self,
        dataframe: pd.DataFrame,
        team_name: str,
        team_type: str,
        season_end_year: int,
        matchweek: int,
        teams_statistics: pd.DataFrame,
    ) -> None:
        """
        Calculate season team points for a specific team in the team_statistics DataFrame based on specified parameters.

        :param dataframe: Input DataFrame containing match data.
        :param team_name: Name of the team to calculate season points for.
        :param team_type: Type of the team ('home_team' or 'away_team').
        :param season_end_year: Ending year of the season.
        :param matchweek: Matchweek number of the game.
        :param teams_statistics: DataFrame to update with calculated statistics.

        :return: None.
        """
        # Adjust season_end_year if matchweek is 1 because the data is empty (beginning of a new season)
        if matchweek == 1:
            season_end_year -= 1

        # Filter the data for the specified team, season, and team type
        team_data: pd.DataFrame = dataframe[
            (
                (dataframe["home_team"] == team_name)
                | (dataframe["away_team"] == team_name)
            )
            & (dataframe["season_end_year"] == season_end_year)
        ]

        season_full_time_points: int = 0
        season_half_time_points: int = 0

        if not team_data.empty:
            # Iterate through the rows of the filtered data
            (
                season_full_time_points,
                season_half_time_points,
            ) = self._calculate_season_points_for_chunk(
                dataframe=team_data, team_name=team_name
            )
        else:
            # If team_data is empty, calculate mean for the three promoted clubs from the previous season

            # Iterate over each promoted club
            for team_name in MatchOutcomePredictor._PROMOTED_CLUBS_2023_24:
                # Filter the data for the specified team, season, and team type
                team_data: pd.DataFrame = dataframe[
                    (
                        (dataframe["home_team"] == team_name)
                        | (dataframe["away_team"] == team_name)
                    )
                    & (dataframe["season_end_year"] == season_end_year)
                ]

                # Calculate full-time and half-time points for the current promoted club's data
                (
                    club_full_time_points,
                    club_half_time_points,
                ) = self._calculate_season_points_for_chunk(
                    dataframe=team_data, team_name=team_name
                )

                # Update cumulative sums with the calculated points for the current promoted club
                season_full_time_points += club_full_time_points / len(
                    MatchOutcomePredictor._PROMOTED_CLUBS_2023_24
                )
                season_half_time_points += club_half_time_points / len(
                    MatchOutcomePredictor._PROMOTED_CLUBS_2023_24
                )

        # Update the team_statistics DataFrame with the calculated season points
        teams_statistics[f"{team_type}_full_time_season_points"] = [
            season_full_time_points
        ]
        teams_statistics[f"{team_type}_half_time_season_points"] = [
            season_half_time_points
        ]

    @staticmethod
    def _calculate_time_since_previous_match_for_prediction(
        dataframe: pd.DataFrame,
        team_name: str,
        team_type: str,
        date: str,
        season_end_year: int,
        teams_statistics: pd.DataFrame,
    ) -> None:
        """
        Calculate the time since the previous match for a specific team in the team_statistics DataFrame
        based on specified parameters.

        :param dataframe: Input DataFrame containing match data.
        :param team_name: Name of the team to calculate time since the previous match for.
        :param team_type: Type of the team ('home_team' or 'away_team').
        :param date: Date of the current match.
        :param season_end_year: Ending year of the season.
        :param teams_statistics: DataFrame to update with calculated statistics.

        :return: None.
        """
        # Filter the data for the specified team, season, and team type
        team_data: pd.DataFrame = dataframe[
            (
                (dataframe["home_team"] == team_name)
                | (dataframe["away_team"] == team_name)
            )
            & (dataframe["season_end_year"] == season_end_year)
        ]

        # Find the last game played by the team in the season
        last_game: Optional[pd.Series] = (
            team_data.iloc[-1] if len(team_data) > 0 else None
        )

        if last_game is not None:
            last_game_date_str: str = last_game["date"]
            last_game_date: datetime.date = pd.to_datetime(
                last_game_date_str, dayfirst=True
            ).date()
            current_date: datetime.date = pd.to_datetime(date, dayfirst=True).date()
            time_since_previous_match: int = (current_date - last_game_date).days
        else:
            time_since_previous_match: np.nan = np.nan

        # Update the team_statistics DataFrame with the calculated time since previous match
        teams_statistics[f"{team_type}_time_since_previous_match"] = [
            time_since_previous_match
        ]

    @staticmethod
    def _calculate_cumulative_sums_for_prediction(
        dataframe: pd.DataFrame,
        team_name: str,
        team_type: str,
        opponent_team_type: str,
        season_end_year: int,
        columns_mapping: List[Tuple[str, str, bool]],
        teams_statistics: pd.DataFrame,
    ) -> None:
        """
        Calculate cumulative sums for specified columns and update team statistics for prediction
        based on specified parameters.

        :param dataframe: Input DataFrame containing match data.
        :param team_name: Name of the team to calculate cumulative sums for.
        :param team_type: Type of team (home_team or away_team).
        :param opponent_team_type: Type of opponent team (away_team or home_team).
        :param season_end_year: Ending year of the season.
        :param columns_mapping: List of tuples containing (data_column, update_column, swap_data).
        :param teams_statistics: DataFrame to update with calculated statistics.

        :return: None.
        """
        # Filter the data for the specified team, team type, and season
        team_data: pd.DataFrame = dataframe[
            (
                (dataframe[team_type] == team_name)
                & (dataframe["season_end_year"] == season_end_year)
            )
        ]

        if (
            team_data.empty
            and team_name not in MatchOutcomePredictor._PROMOTED_CLUBS_2024_25
        ):
            team_data: pd.DataFrame = dataframe[
                (
                    (dataframe[team_type] == team_name)
                    & (dataframe["season_end_year"] == season_end_year - 1)
                )
            ]

        # Create a dictionary to store cumulative sums
        cumulative_sums: Dict[str, List[float]] = {
            update_column: [0, 0] for _, update_column, swap_data in columns_mapping
        }

        if not team_data.empty:
            for _, row in team_data.iterrows():
                for data_column, update_column, swap_data in columns_mapping:
                    cumulative_key: str = (
                        f"{opponent_team_type}_{data_column}"
                        if swap_data
                        else f"{team_type}_{data_column}"
                    )

                    # Update the cumulative sum for the current data column
                    cumulative_sums[update_column][0] += row[cumulative_key]
                    cumulative_sums[update_column][1] += 1
        else:
            # If team_data is empty, calculate mean for the three promoted clubs from the previous season
            promoted_clubs_data: pd.DataFrame = dataframe[
                (
                    dataframe[team_type].isin(
                        MatchOutcomePredictor._PROMOTED_CLUBS_2023_24
                    )
                )
                & (dataframe["season_end_year"] == 2024)
            ]

            chunk_length_per_club: int = len(
                MatchOutcomePredictor._PROMOTED_CLUBS_2023_24
            )

            for i in range(0, len(promoted_clubs_data), chunk_length_per_club):
                chunk: pd.DataFrame = promoted_clubs_data.iloc[
                    i : i + chunk_length_per_club
                ]

                for data_column, update_column, swap_data in columns_mapping:
                    cumulative_key = (
                        f"{opponent_team_type}_{data_column}"
                        if swap_data
                        else f"{team_type}_{data_column}"
                    )

                    # Update the cumulative sum for the current data column
                    cumulative_sums[update_column][0] += (
                        sum(chunk[cumulative_key]) / chunk_length_per_club
                    )
                    cumulative_sums[update_column][1] += 1

        for _, update_column, _ in columns_mapping:
            # Calculate the cumulative average for the data column
            if cumulative_sums[update_column][1] == 0:
                cumulative_average: np.nan = np.nan
            else:
                cumulative_average: float = (
                    cumulative_sums[update_column][0]
                    / cumulative_sums[update_column][1]
                )

            # Update the team_statistics DataFrame with the calculated cumulative average
            teams_statistics[f"{team_type}_cumulative_{update_column}"] = [
                cumulative_average
            ]

    def _validate_predict_input_parameters(
        self,
        home_team: str,
        away_team: str,
        matchweek: int,
        season_end_year: int,
        date: str,
    ) -> None:
        """
        Validates the input parameters for the predict method.

        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :param matchweek: Matchweek number of the game.
        :param season_end_year: Ending year of the season in which the game takes place.
        :param date: Date of the game in string format (DD/MM/YYYY).
        :return: None

        :raises ValueError: If any of the input parameters are invalid.
        """
        # Validate home_team and away_team
        if not isinstance(home_team, str) or not home_team.replace(" ", "").isalnum():
            raise ValueError(
                f"Invalid home_team name: {home_team}. It should be a non-empty alphanumeric string."
            )

        if not isinstance(away_team, str) or not away_team.replace(" ", "").isalnum():
            raise ValueError(
                f"Invalid away_team name: {away_team}. It should be a non-empty alphanumeric string."
            )

        # Check that away_team is different from home_team
        if away_team == home_team:
            raise ValueError(
                f"The away_team ({away_team}) must be different from the home_team ({home_team})."
            )

        # Validate matchweek
        if not (isinstance(matchweek, int) and 1 <= matchweek <= 38):
            raise ValueError(
                f"Invalid matchweek: {matchweek}. It should be an integer between 1 and 38."
            )

        # Validate season_end_year
        if not isinstance(season_end_year, int) or season_end_year <= 2024:
            raise ValueError(
                f"Invalid season_end_year: {season_end_year}. It should be an integer greater than 2024."
            )

        try:
            # Validate date and check if year is 2025 or later
            parsed_date: datetime.date = pd.to_datetime(date)
        except ValueError:
            raise ValueError(
                "Invalid date format. It should be in the format DD/MM/YYYY."
            )
        if parsed_date.year < self._SEASON_END_YEAR - 1:
            raise ValueError(
                f"Invalid date. Year ({parsed_date.year}) must be 2024 or later."
            )

    def _predict(
        self,
        home_team: str,
        away_team: str,
        matchweek: int,
        season_end_year: int,
        date: str,
    ) -> str:
        """
        Predicts the outcome of a new game based on the provided input parameters.

        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :param matchweek: Matchweek number of the game.
        :param season_end_year: Ending year of the season in which the game takes place.
        :param date: Date of the game in string format (DD/MM/YYYY).

        :return: Predicted outcome of the game as a string: 'H' for 'Home Team Wins', 'D' for 'Draw', 'A' for 'Away
        Team Wins'.

        :raises IOError: If an error occurs while loading the processed DataFrame or trained model.
        :raises joblib.pickle.PicklingError: If a pickling error occurs while loading the trained model.
        """
        # Validate input parameters using the dedicated validation method
        self._validate_predict_input_parameters(
            home_team, away_team, matchweek, season_end_year, date
        )

        try:
            # Load the processed DataFrame from the saved file
            processed_data_path: str = os.path.join(
                self._SAVED_MODEL_DIRECTORY, "processed_dataframe.csv"
            )
            processed_dataframe: pd.DataFrame = pd.read_csv(processed_data_path)

            # Load the trained model from the saved file
            trained_model_path: str = os.path.join(
                self._SAVED_MODEL_DIRECTORY, "trained_model.pkl"
            )
            trained_model: Union[XGBClassifier, CatBoostClassifier, LGBMClassifier] = (
                joblib.load(trained_model_path)
            )
        except (IOError, joblib.pickle.PicklingError) as e:
            raise IOError(
                f"An error occurred while loading the processed DataFrame or trained model: {e}"
            )

        # Create a DataFrame to store teams' statistics
        teams_statistics: pd.DataFrame = pd.DataFrame()

        # Dictionary to map team index to team type
        team_index_to_type: Dict[int, str] = {0: "home_team", 1: "away_team"}

        # Loop through both home and away teams
        for i, team_name in enumerate([home_team, away_team]):
            # Loop through different numbers of last games
            for last_games in MatchOutcomePredictor._LAST_GAMES_LIST:
                # Update team statistics based on the last games played
                self._update_team_statistics_for_prediction(
                    dataframe=processed_dataframe,
                    team_name=team_name,
                    team_type=team_index_to_type[i],
                    opponent_team_type=team_index_to_type[1 - i],
                    season_end_year=season_end_year,
                    last_games=last_games,
                    columns_mapping=[
                        ("full_time_game_points", "full_time_points_scored", False),
                        ("full_time_game_points", "full_time_points_conceded", True),
                        ("half_time_game_points", "half_time_points_scored", False),
                        ("half_time_game_points", "half_time_points_conceded", True),
                        ("full_time_goals", "full_time_goals_scored", False),
                        ("full_time_goals", "full_time_goals_conceded", True),
                        ("half_time_goals", "half_time_goals_scored", False),
                        ("half_time_goals", "half_time_goals_conceded", True),
                        ("shots", "shots_taken", False),
                        ("shots", "shots_faced", True),
                        ("shots_on_target", "shots_on_target_taken", False),
                        ("shots_on_target", "shots_on_target_faced", True),
                        ("corners", "corners_won", False),
                        ("corners", "corners_conceded", True),
                        ("fouls", "fouls_committed", False),
                        ("fouls", "fouls_received", True),
                        ("yellow_cards", "yellow_cards_given", False),
                        ("yellow_cards", "yellow_cards_given_to_opponent", True),
                        ("red_cards", "red_cards_given", False),
                        ("red_cards", "red_cards_given_to_opponent", True),
                    ],
                    teams_statistics=teams_statistics,
                )

                # Update head-to-head score statistics
                self._update_head_to_head_score_for_prediction(
                    dataframe=processed_dataframe,
                    team_name=team_name,
                    opponent_name=away_team if team_name == home_team else home_team,
                    team_type=team_index_to_type[i],
                    opponent_team_type=team_index_to_type[1 - i],
                    season_end_year=season_end_year,
                    last_games=last_games,
                    teams_statistics=teams_statistics,
                )

            # Calculate season team points statistics
            self._calculate_season_team_points_for_prediction(
                dataframe=processed_dataframe,
                team_name=team_name,
                team_type=team_index_to_type[i],
                season_end_year=season_end_year,
                matchweek=matchweek,
                teams_statistics=teams_statistics,
            )

            # Calculate time since previous match statistics
            self._calculate_time_since_previous_match_for_prediction(
                dataframe=processed_dataframe,
                team_name=team_name,
                team_type=team_index_to_type[i],
                date=date,
                season_end_year=season_end_year,
                teams_statistics=teams_statistics,
            )

            # Calculate cumulative sums statistics
            self._calculate_cumulative_sums_for_prediction(
                dataframe=processed_dataframe,
                team_name=team_name,
                team_type=team_index_to_type[i],
                opponent_team_type=team_index_to_type[1 - i],
                season_end_year=season_end_year,
                columns_mapping=[
                    ("full_time_game_points", "full_time_points_scored", False),
                    ("full_time_game_points", "full_time_points_conceded", True),
                    ("half_time_game_points", "half_time_points_scored", False),
                    ("half_time_game_points", "half_time_points_conceded", True),
                    ("full_time_goals", "full_time_goals_scored", False),
                    ("full_time_goals", "full_time_goals_conceded", True),
                    ("half_time_goals", "half_time_goals_scored", False),
                    ("half_time_goals", "half_time_goals_conceded", True),
                    ("shots", "shots_taken", False),
                    ("shots", "shots_faced", True),
                    ("shots_on_target", "shots_on_target_taken", False),
                    ("shots_on_target", "shots_on_target_faced", True),
                    ("corners", "corners_won", False),
                    ("corners", "corners_conceded", True),
                    ("fouls", "fouls_committed", False),
                    ("fouls", "fouls_received", True),
                    ("yellow_cards", "yellow_cards_given", False),
                    ("yellow_cards", "yellow_cards_given_to_opponent", True),
                    ("red_cards", "red_cards_given", False),
                    ("red_cards", "red_cards_given_to_opponent", True),
                ],
                teams_statistics=teams_statistics,
            )

        # Calculate feature differences for all statistics
        statistics_dataframe: pd.DataFrame = self._calculate_all_feature_differences(
            dataframe=teams_statistics
        )

        # Adding season_end_year to the statistics DataFrame
        statistics_dataframe["season_end_year"] = [season_end_year]

        # Assign the numeric representation of the season
        statistics_dataframe["season"] = [self._get_season(date=pd.to_datetime(date))]

        # Adding matchweek to the statistics DataFrame
        statistics_dataframe["matchweek"] = [matchweek]

        # Use the best model to perform prediction based on calculated features
        prediction_numeric = trained_model.predict(statistics_dataframe)

        # Convert numeric prediction to corresponding label
        prediction_label = (
            "H" if prediction_numeric == 2 else "D" if prediction_numeric == 1 else "A"
        )

        return prediction_label

    def update_predicted_fixtures(self) -> None:
        """
        Updates the predicted outcomes for fixtures in the database.

        :return: None
        """
        # Get all fixtures from the database, sorted by date and time
        fixtures: QuerySet[Fixture] = Fixture.objects.order_by("date", "time")

        # Filter fixtures without predicted outcomes
        fixtures_without_predicted_outcomes: List[Fixture] = [
            fixture for fixture in fixtures if fixture.predicted_outcome is None
        ]

        # Initialize matchweek with the matchweek from the first fixture (if available)
        matchweek: Optional[IntegerField] = (
            fixtures_without_predicted_outcomes[0].matchweek
            if fixtures_without_predicted_outcomes
            else None
        )

        # Iterate through fixtures
        for fixture in fixtures_without_predicted_outcomes:
            # Check if the matchweek is different from the current fixture's matchweek
            if matchweek is not None and matchweek != fixture.matchweek:
                return  # Skip further prediction if matchweek is different

            home_team: str = fixture.home_team.name
            away_team: str = fixture.away_team.name

            # Format the fixture date as DD/MM/YYYY
            formatted_date: str = fixture.date.strftime("%d/%m/%Y")

            # Predict the outcome for the fixture
            predicted_outcome: str = self._predict(
                home_team=home_team,
                away_team=away_team,
                matchweek=fixture.matchweek,
                season_end_year=fixture.season_end_year,
                date=formatted_date,
            )

            # Update the predicted outcome for the fixture
            fixture.predicted_outcome = predicted_outcome
            fixture.save()

    def combine_and_save_dataframes(
        self, processed_games: List[Dict[str, Any]]
    ) -> None:
        """
        Processes the list of processed games, converts them to a DataFrame, and appends them to an existing CSV file.

        :param processed_games: A list of dictionaries containing processed game data.
        :return: None.
        """
        # Read the row dataframe CSV file as a DataFrame
        row_dataframe: pd.DataFrame = pd.read_csv(self._INPUT_DATA_PATH)

        # Convert processed_games to a DataFrame
        processed_games_dataframe: pd.DataFrame = pd.DataFrame(processed_games)

        # Concatenate the two DataFrames (add rows)
        combined_dataframe: pd.DataFrame = pd.concat(
            [row_dataframe, processed_games_dataframe], ignore_index=True
        )

        # Save the combined DataFrame to the same path
        combined_dataframe.to_csv(self._INPUT_DATA_PATH, index=False, na_rep="NaN")
