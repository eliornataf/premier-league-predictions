import re

import numpy as np
from datetime import datetime, timedelta
from django.core.exceptions import ObjectDoesNotExist
from selenium import webdriver
from selenium.webdriver.safari.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from typing import Optional, List, Tuple, Dict, Any
from bs4 import BeautifulSoup, ResultSet, PageElement, Tag
from selenium.common.exceptions import TimeoutException
from website.models import FootballClub, Fixture, Result, TableRow


class FootballMatchScraper:
    """
    A class for scraping football match fixtures, results, and Premier League table from web sources.

    This class is designed to retrieve data related to football match fixtures, results, and Premier League table
    from websites. It provides methods to extract, parse, and process information about upcoming fixtures,
    match results, and current Premier League standings.

    Attributes:
      _FIXTURES_URL (str): The URL for fetching football match fixture data.
      _RESULTS_URL (str): The URL for fetching football match result data.
      _TABLE_URL (str): The URL for fetching Premier League table data.
      _SEASON_END_YEAR (int): The year in which the current football season ends.

    Methods:
      process_fixtures_html(): Extracts and processes football match fixture data.
      process_results_html(last_results_update_date): Extracts and processes football match result data.
      process_table_html(): Scrapes and processes the Premier League table data from the specified URL.

    """

    _FIXTURES_URL: str = "https://www.skysports.com/premier-league-fixtures"
    _RESULTS_URL: str = "https://www.skysports.com/premier-league-results"
    _TABLE_URL: str = "https://www.skysports.com/premier-league-table"
    _SEASON_END_YEAR: int = 2025

    @staticmethod
    def _get_html_from_url(url: str) -> str:
        """
        Loads the given URL using Selenium, accepts a cookie message, and returns the HTML of the page.

        :param url: The URL to load and retrieve HTML from.
        :return: The HTML content of the loaded page.

        :raises Exception: If there's an error during the page loading and manipulation process.
        """
        # Initialize the driver variable
        driver: Optional[WebDriver] = None

        try:
            driver = webdriver.Safari()

            # Load the URL
            driver.get(url)

            # Locate and switch to the iframe containing the cookie message
            iframe = WebDriverWait(driver, 10).until(
                expected_conditions.presence_of_element_located(
                    (By.XPATH, '//*[@id="sp_message_iframe_1168576"]')
                )
            )
            driver.switch_to.frame(iframe)

            # Find and click the accept button for the cookie message
            accept_button = WebDriverWait(driver, 10).until(
                expected_conditions.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "button[title='Accept all'][aria-label='Accept all']",
                    )
                )
            )
            accept_button.click()

            # Switch back to the default content
            driver.switch_to.default_content()

            # Check if the "plus-more" button is present
            try:
                plus_more_button = WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located(
                        (By.CLASS_NAME, "plus-more")
                    )
                )
                plus_more_button.click()
            except TimeoutException:
                pass  # "plus-more" button not found, continue without clicking

            # Return the HTML of the page
            html: str = driver.page_source

        except Exception as e:
            raise Exception(f"An error occurred while fetching the HTML content: {e}")

        finally:
            # Close the Selenium webdriver if it's not None
            if driver is not None:
                driver.quit()

        return html

    @staticmethod
    def _parse_child_elements(html_content: str) -> List[PageElement]:
        """
        Parses the HTML content using BeautifulSoup and extracts the direct child elements of a specific class.

        :param html_content: The HTML content to parse.
        :return: List of direct child elements.

        :raises Exception: If no child elements are found in the HTML or if there's an error during parsing.
        """
        # Parse the HTML content using BeautifulSoup
        try:
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
        except Exception as parsing_error:
            raise Exception(f"Error parsing HTML content: {parsing_error}")

        # Get the direct child elements of the fixtures or results body
        child_elements: List[PageElement] = soup.find(class_="fixres__body").contents

        if child_elements is None:
            raise Exception("No child elements found in the HTML")

        return child_elements

    @staticmethod
    def _extract_game_year(element: PageElement) -> int:
        """
        Extracts the game year from the text content of an HTML PageElement and returns its integer representation.

        :param element: HTML PageElement containing text.
        :return: The integer representation of the game year.
        :raises ValueError: If the input element is not a Tag, if the extracted last word is not a valid integer, or
                            if the input element's text content does not contain exactly two words.
        """
        text: str = element.text
        if text is None:
            raise ValueError("Input element's text content is None.")

        words: List[str] = text.split()
        if len(words) != 2:
            raise ValueError(
                f"Input element's text content must contain exactly two words, but found {len(words)} words."
            )

        last_word: str = words[-1]
        try:
            year = int(last_word)
            return year
        except ValueError:
            raise ValueError("Last word is not a valid integer.")

    @staticmethod
    def _parse_clubs(element: PageElement) -> Tuple[str, str]:
        """
        Parse the home and away teams from the provided HTML PageElement.

        :param element: HTML PageElement containing match information.
        :return: A list containing the names of the home and away teams.
        :raises ValueError: If an invalid number of clubs is found.
        """
        clubs: ResultSet = element.find_all(class_="swap-text__target")

        # Check if there are at least 2 elements (2 clubs) and handle the case where there's a third element (text about
        # bets)
        if len(clubs) < 2:
            raise ValueError(
                f"Expected at least 2 elements (2 clubs), but found {len(clubs)} elements."
            )

        home_team: str = clubs[0].text
        away_team: str = clubs[1].text

        return home_team, away_team

    @staticmethod
    def _convert_time_format(match_time: datetime) -> datetime.time:
        """
        Converts a match time datetime object to "HH:MM:SS" in the Israel timezone.

        :param match_time: Match time as a datetime object.
        :return: Match time as a datetime.time object in the Israel timezone.

        :raises ValueError: If the conversion encounters an error.
        """
        try:
            # Calculate the time difference between UK and Israel timezones
            uk_time_difference: timedelta = timedelta(hours=0)
            israel_time_difference: timedelta = timedelta(hours=2)

            # Apply the time difference to the match time
            israel_match_time: datetime = match_time + (
                israel_time_difference - uk_time_difference
            )

            # Return the time as a datetime.time object
            return israel_match_time.time()
        except (ValueError, TypeError):
            raise ValueError("An error occurred while converting the match time.")

    def _parse_match_time(self, element: PageElement) -> datetime.time:
        """
        Parse the match time from the provided HTML PageElement.

        :param element: HTML PageElement containing match information.
        :return: The cleaned match time as a datetime.time object.

        :raises ValueError: If there's an error during the information extraction.
        """
        try:
            # Extract match time element
            time_element: Tag = element.find(class_="matches__date")

            # Extract and clean match time
            time_str: str = time_element.get_text(strip=True)
            if not time_str:
                raise ValueError("Match time is missing.")

            # Parse the match time from the provided HTML
            match_time: datetime = datetime.strptime(time_str, "%H:%M")

            # Convert and return the time in the Israel timezone
            israel_time: datetime.time = self._convert_time_format(match_time)
            return israel_time
        except Exception as e:
            raise ValueError(f"An error occurred while parsing match time: {e}.")

    @staticmethod
    def _convert_date_format(element: PageElement, year: int) -> datetime.date:
        """
        Converts a date PageElement's text to a datetime.date object with the specified year.

        :param element: Date PageElement containing the date in the format "%A %d %B".
        :param year: The year to use in the resulting datetime.date object.
        :return: datetime.date object with the specified year.

        :raises ValueError: If the input date is empty or if the conversion encounters an error.
        """
        date_str: str = element.text
        if not date_str:
            raise ValueError("Input date PageElement must not be empty.")

        try:
            # Remove the ordinal suffix from the day number
            day_number: int = int(date_str.split()[1][:-2])

            # Remove the ordinal suffix from the date string
            date_str_without_suffix: str = date_str.replace(
                date_str.split()[1], str(day_number)
            )

            # Parse the modified date string using strptime
            parsed_date: datetime = datetime.strptime(
                date_str_without_suffix, "%A %d %B"
            )

            # Replace the year with the specified value
            modified_date: datetime = parsed_date.replace(year=year)

            # Extract the date part as a datetime.date object
            modified_date_date: datetime.date = modified_date.date()

            return modified_date_date
        except (ValueError, TypeError):
            raise ValueError("An error occurred while converting the date PageElement.")

    @staticmethod
    def _add_game_to_database(
        home_team: str,
        away_team: str,
        date: datetime.date,
        time: datetime.time,
        matchweek: int,
        season_end_year: int,
    ) -> None:
        """
        Add a football game fixture to the database.

        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :param date: Match date as a datetime.date object.
        :param time: Match time as a datetime.time object.
        :param matchweek: Matchweek number.
        :param season_end_year: The year in which the current football season ends.
        :return: None.

        :raises ObjectDoesNotExist: If either home_team or away_team does not exist in the database.
        :raises Exception: If an error occurs while adding the game to the database.
        """
        try:
            # Get instances of home and away teams from the database
            home_team_instance = FootballClub.objects.get(name=home_team)
            away_team_instance = FootballClub.objects.get(name=away_team)

            # Create a new Fixture instance and add it to the database
            Fixture.objects.create(
                home_team=home_team_instance,
                away_team=away_team_instance,
                date=date,
                time=time,
                matchweek=matchweek,
                season_end_year=season_end_year,
            )
        except ObjectDoesNotExist as e:
            raise ObjectDoesNotExist(
                f"One or both of the teams does not exist in the database: {e}."
            )
        except Exception as e:
            raise Exception(
                f"An error occurred while adding the game to the database: {e}"
            )

    def process_fixtures_html(self) -> None:
        """
        Extracts relevant information from the fixtures HTML content and processes fixtures.

        :return: None.
        """
        # Get HTML content from fixtures URL
        html_content: str = self._get_html_from_url(url=self._FIXTURES_URL)

        # Parse child elements
        child_elements: List[PageElement] = self._parse_child_elements(
            html_content=html_content
        )

        last_updated_year: Optional[int] = None
        last_updated_date: Optional[datetime.date] = None

        # Initialize games_in_current_matchweek and matchweek_number
        games_in_current_matchweek: int = 0
        matchweek_number: int = 1

        for element in child_elements:
            element_tag_name: str = element.name
            if element_tag_name == "h3":
                # Extract and update the last updated year
                last_updated_year = self._extract_game_year(element=element)
            elif element_tag_name == "h4":
                # Extract and update the last updated date
                last_updated_date = self._convert_date_format(
                    element=element, year=last_updated_year
                )
            elif element_tag_name == "div":
                home_team, away_team = self._parse_clubs(element=element)
                time: datetime.time = self._parse_match_time(element=element)

                # Check if games_in_current_matchweek exceeds 10 (a matchweek includes 10 games)
                if games_in_current_matchweek >= 10:
                    games_in_current_matchweek = 0
                    matchweek_number += 1

                self._add_game_to_database(
                    home_team=home_team,
                    away_team=away_team,
                    date=last_updated_date,
                    time=time,
                    matchweek=matchweek_number,
                    season_end_year=self._SEASON_END_YEAR,
                )

                # Increment games_in_current_matchweek
                games_in_current_matchweek += 1

    @staticmethod
    def _parse_team_scores(element: PageElement) -> Tuple[int, int]:
        """
        Parse home and away team scores from the given HTML element.

        :param element: The HTML element containing team scores.
        :return: A tuple containing home team's score and away team's actual score as integers.

        :raises ValueError: If an error occurs while parsing team scores or the number of clubs is invalid.
        """
        team_scores: ResultSet = element.find_all(class_="matches__teamscores-side")

        # Check if the number of team_scores is exactly 2
        if len(team_scores) != 2:
            raise ValueError(
                f"Invalid number of clubs. Found {len(team_scores)} team scores instead of 2."
            )

        try:
            home_team_score = int(team_scores[0].text)
            away_team_score = int(team_scores[1].text)
        except ValueError:
            raise ValueError("Invalid score format: Scores should be integers.")

        return home_team_score, away_team_score

    @staticmethod
    def _retrieve_matching_fixture(home_team: str, away_team: str) -> Fixture:
        """
        Retrieve a fixture that matches the provided home and away teams.

        :param home_team: The name of the home team.
        :param away_team: The name of the away team.
        :return: The matching fixture object.

        :raises Exception: If the number of matching fixtures is not exactly 1.
        """
        # Query the database for fixtures that match the provided home and away teams
        matching_fixtures = Fixture.objects.filter(
            home_team__name=home_team, away_team__name=away_team
        )

        # Check if exactly one matching fixture was found
        if matching_fixtures.count() != 1:
            raise Exception(
                f"Invalid number of matching fixtures. Found {matching_fixtures.count()} instead of 1."
            )

        # Retrieve the first (and only) matching fixture
        fixture: Fixture = matching_fixtures.first()

        return fixture

    @staticmethod
    def _determine_full_time_result(home_team_score: int, away_team_score: int) -> str:
        """
        Determine the full-time result based on goals.

        :param home_team_score: Score of the home team.
        :param away_team_score: Score of the away team.
        :return: Full-time result ('H' for home win, 'A' for away win, 'D' for draw).
        """
        if home_team_score > away_team_score:
            return "H"
        elif home_team_score < away_team_score:
            return "A"
        else:
            return "D"

    @staticmethod
    def _process_game_statistics(soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Process statistics elements from the game statistics page.

        :param soup: BeautifulSoup object representing the game statistics page.
        :return: A dictionary containing the processed statistics.

        :raises ValueError: If the number of extracted statistics is not 32 or if there are conversion errors.
        """
        stats_elements: ResultSet = soup.find_all(class_="sdc-site-match-stats__val")

        if len(stats_elements) != 32:
            raise ValueError(
                f"Expected 32 statistics on the page, but found {len(stats_elements)}."
            )

        processed_statistics: Dict[str, Any] = {}

        try:
            processed_statistics["home_team_possession"] = float(stats_elements[0].text)
            processed_statistics["away_team_possession"] = float(stats_elements[1].text)
            processed_statistics["home_team_shots"] = int(stats_elements[2].text)
            processed_statistics["away_team_shots"] = int(stats_elements[3].text)
            processed_statistics["home_team_shots_on_target"] = int(
                stats_elements[4].text
            )
            processed_statistics["away_team_shots_on_target"] = int(
                stats_elements[5].text
            )
            processed_statistics["home_team_shots_off_target"] = int(
                stats_elements[6].text
            )
            processed_statistics["away_team_shots_off_target"] = int(
                stats_elements[7].text
            )
            processed_statistics["home_team_shots_blocked"] = int(
                stats_elements[8].text
            )
            processed_statistics["away_team_shots_blocked"] = int(
                stats_elements[9].text
            )
            processed_statistics["home_team_passing"] = float(stats_elements[10].text)
            processed_statistics["away_team_passing"] = float(stats_elements[11].text)
            processed_statistics["home_team_clear_cut_chances"] = int(
                stats_elements[12].text
            )
            processed_statistics["away_team_clear_cut_chances"] = int(
                stats_elements[13].text
            )
            processed_statistics["home_team_corners"] = int(stats_elements[14].text)
            processed_statistics["away_team_corners"] = int(stats_elements[15].text)
            processed_statistics["home_team_offsides"] = int(stats_elements[16].text)
            processed_statistics["away_team_offsides"] = int(stats_elements[17].text)
            processed_statistics["home_team_tackles"] = float(stats_elements[18].text)
            processed_statistics["away_team_tackles"] = float(stats_elements[19].text)
            processed_statistics["home_team_aerial_duels"] = float(
                stats_elements[20].text
            )
            processed_statistics["away_team_aerial_duels"] = float(
                stats_elements[21].text
            )
            processed_statistics["home_team_fouls_committed"] = int(
                stats_elements[24].text
            )
            processed_statistics["away_team_fouls_committed"] = int(
                stats_elements[25].text
            )
            processed_statistics["home_team_fouls_won"] = int(stats_elements[26].text)
            processed_statistics["away_team_fouls_won"] = int(stats_elements[27].text)
            processed_statistics["home_team_yellow_cards"] = int(
                stats_elements[28].text
            )
            processed_statistics["away_team_yellow_cards"] = int(
                stats_elements[29].text
            )
            processed_statistics["home_team_red_cards"] = int(stats_elements[30].text)
            processed_statistics["away_team_red_cards"] = int(stats_elements[31].text)
        except (ValueError, IndexError) as conversion_error:
            raise ValueError(f"Error converting statistics: {conversion_error}")

        return processed_statistics

    def _process_game_stats_url(self, url_parts: List[str]) -> Dict[str, Any]:
        """
        Process game statistics by modifying the URL, fetching HTML content, parsing it.

        :param url_parts: List of URL parts to construct the modified URL.
        :return: A dictionary containing the processed statistics.

        :raises Exception: If there's an error fetching or parsing the HTML content.
        """
        # Modify the URL by adding '/stats/' to the base URL to access game statistics
        modified_url: str = url_parts[0] + "/stats/" + url_parts[1]

        # Send the modified URL to the _get_html_from_url method to fetch HTML content
        html_content: str = self._get_html_from_url(url=modified_url)

        try:
            # Parse the HTML content using Beautiful Soup
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
        except Exception as parsing_error:
            raise Exception(f"Error parsing HTML content: {parsing_error}")

        # Process game statistics from the game statistics page
        processed_statistics: Dict[str, Any] = self._process_game_statistics(soup=soup)

        return processed_statistics

    @staticmethod
    def _format_team_goal_scorers(team_goal_scorers: Dict[str, List[int]]) -> str:
        """
        Format and sort goal scorers for a team.

        :param team_goal_scorers: A dictionary containing player surnames as keys and lists of goal times as values.
        :return: A formatted string of goal scorers for the team.
        """
        # Sort players by their goal times
        sorted_players: List[Tuple[str, List[int]]] = sorted(
            team_goal_scorers.items(), key=lambda x: x[1]
        )

        # Format each player's goal scorers
        formatted_scorers: List[str] = []
        for player, goals in sorted_players:
            if len(goals) == 1:
                formatted_scorers.append(f"{player} ({goals[0]})")
            else:
                formatted_scorers.append(f"{player} ({', '.join(map(str, goals))})")

        # Join formatted scorers into a single string
        return ", ".join(formatted_scorers)

    def _parse_goal_scorers(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """
        Parse and format goal scorers from the soup object.

        :param soup: BeautifulSoup object containing the parsed HTML content.
        :return: A tuple containing two formatted strings: home_team_goal_scorers and away_team_goal_scorers.

        :raises Exception: If the number of team_events is not 2.
        """
        # Find all "sdc-site-team-lineup__col" elements by class
        team_events: ResultSet = soup.find_all(class_="sdc-site-team-lineup__col")

        # Check if the number of team_events is not equal to 2
        if len(team_events) != 2:
            raise Exception(
                f"Expected 2 team events for home and away teams, but got {len(team_events)}."
            )

        # Initialize a list of dictionaries to store goal scorers for both teams.
        team_goal_scorers: List[Dict[str, List[int]]] = [{} for _ in range(2)]

        # Loop through the home team events (team_events[0]) and away team events (team_events[1])
        for i, team_event in enumerate(team_events):
            # Find all "sdc-site-team-lineup__events" elements by class
            player_events: ResultSet = team_event.find_all(
                class_="sdc-site-team-lineup__events"
            )

            # Loop through player events
            for player_event in player_events:
                # Loop through events of player_event
                for event in player_event:
                    # Check if the event contains "Goal scored" or "Penalty scored"
                    if "Goal scored" in event.text or "Penalty scored" in event.text:
                        # Extract player surname who scored the goal
                        player_surname: str = player_event.parent.find(
                            class_="sdc-site-team-lineup__player-surname"
                        ).text

                        # Extract goal time and remove single quotes
                        goal_time: str = event.find(
                            class_="sdc-site-team-lineup__event_time"
                        ).text.replace("'", "")

                        # Check if goal time contains a '+' and reconstruct goal_time
                        if "+" in goal_time:
                            parts: List[str] = goal_time.split("+")
                            goal_time: str = parts[0][0] + parts[1]

                        # Add the goal scorer to the team_goal_scorers dictionary for the current team
                        if player_surname in team_goal_scorers[i]:
                            team_goal_scorers[i][player_surname].append(int(goal_time))
                        else:
                            team_goal_scorers[i][player_surname] = [int(goal_time)]

        # Format and sort the goal scorers for home and away teams
        home_team_goal_scorers: str = self._format_team_goal_scorers(
            team_goal_scorers=team_goal_scorers[0]
        )
        away_team_goal_scorers: str = self._format_team_goal_scorers(
            team_goal_scorers=team_goal_scorers[1]
        )

        return home_team_goal_scorers, away_team_goal_scorers

    def _process_goal_scorers(
        self,
        url_parts: List[str],
    ) -> Dict[str, str]:
        """
        Process game goal scorers by modifying the URL, fetching HTML content, parsing it, and extracting goal scorers.

        :param url_parts: List of URL parts to construct the modified URL.
        :return: A dictionary containing the home_team_goal_scorers and away_team_goal_scorers as strings.

        :raises Exception: If there's an error fetching, parsing the HTML content, or if the number of results is not 2.
        """
        # Create modified URL by adding "teams" between url_parts[0] and url_parts[1]
        modified_url: str = url_parts[0] + "/teams/" + url_parts[1]

        # Send the modified URL to the _get_html_from_url method to fetch HTML content
        html_content: str = self._get_html_from_url(url=modified_url)

        try:
            # Parse the HTML content using Beautiful Soup
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
        except Exception as parsing_error:
            raise Exception(f"Error parsing HTML content: {parsing_error}")

        # Parse goal scorers using _parse_goal_scorers method
        home_team_goal_scorers, away_team_goal_scorers = self._parse_goal_scorers(soup)

        return {
            "home_team_goal_scorers": home_team_goal_scorers,
            "away_team_goal_scorers": away_team_goal_scorers,
        }

    def _process_game_detailed_data(
        self,
        element: PageElement,
    ) -> Dict[str, Any]:
        """
        Process detailed data for a game element.

        This method extracts the link to the detailed page for the game and returns the processed statistics.

        :param element: A BeautifulSoup PageElement representing a game.
        :return: A dictionary containing the game detailed data.

        :raises Exception: If the link tag is not found in the element and if there's an error parsing the HTML content.
        """
        # Extract the link from the element
        link_tag: Tag = element.find("a", class_="matches__item matches__link")

        if not link_tag:
            raise Exception("Link tag not found in the element")

        # Extract the 'href' attribute from the link tag to get the detailed page URL
        url: str = link_tag["href"]

        # Split the URL by the last '/' to separate the base URL
        url_parts: List[str] = url.rsplit("/", 1)

        # Call _process_game_stats_url with url_parts
        processed_statistics: Dict[str, Any] = self._process_game_stats_url(url_parts)

        # Call _process_goal_scorers and save its result to a variable
        goal_scorers: Dict[str, str] = self._process_goal_scorers(url_parts)

        # Combine processed_statistics and goal_scorers into a single dictionary
        game_detailed_data: Dict[str, Any] = {**processed_statistics, **goal_scorers}

        return game_detailed_data

    def _create_row_data(
        self,
        season_end_year: int,
        formatted_date: str,
        home_team: str,
        away_team: str,
        home_team_score: int,
        away_team_score: int,
        game_detailed_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create row_data dictionary for the row dataframe.

        :param season_end_year: End year of the season.
        :param formatted_date: Formatted date of the fixture (DD/MM/YYYY).
        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :param home_team_score: Score of the home team.
        :param away_team_score: Score of the away team.
        :param game_detailed_data: The game detailed data.
        :return: Dictionary containing row data.
        """
        row_data: Dict[str, Any] = {
            "season_end_year": season_end_year,
            "date": formatted_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_full_time_goals": home_team_score,
            "away_team_full_time_goals": away_team_score,
            "full_time_result": self._determine_full_time_result(
                home_team_score=home_team_score, away_team_score=away_team_score
            ),
            "home_team_half_time_goals": np.nan,
            "away_team_half_time_goals": np.nan,
            "half_time_result": np.nan,
            "home_team_shots": game_detailed_data["home_team_shots"],
            "away_team_shots": game_detailed_data["away_team_shots"],
            "home_team_shots_on_target": game_detailed_data[
                "home_team_shots_on_target"
            ],
            "away_team_shots_on_target": game_detailed_data[
                "away_team_shots_on_target"
            ],
            "home_team_corners": game_detailed_data["home_team_corners"],
            "away_team_corners": game_detailed_data["away_team_corners"],
            "home_team_fouls": game_detailed_data["home_team_fouls_committed"],
            "away_team_fouls": game_detailed_data["away_team_fouls_committed"],
            "home_team_yellow_cards": game_detailed_data["home_team_yellow_cards"],
            "away_team_yellow_cards": game_detailed_data["away_team_yellow_cards"],
            "home_team_red_cards": game_detailed_data["home_team_red_cards"],
            "away_team_red_cards": game_detailed_data["away_team_red_cards"],
        }

        return row_data

    def process_results_html(self) -> List[Dict[str, Any]]:
        """
        Extracts relevant information from the results HTML content and processes results.

        :return: A list containing the row data dictionaries for each processed game.
        """
        # Get HTML content from results URL
        html_content: str = self._get_html_from_url(url=self._RESULTS_URL)

        # Parse child elements
        child_elements: List[PageElement] = self._parse_child_elements(
            html_content=html_content
        )

        # Reverse the order of child_elements
        reversed_child_elements = reversed(child_elements)

        # Initialize matchweek variable to keep track of the current matchweek being processed.
        matchweek: Optional[int] = None

        # List to store row data dictionaries for each game
        data_rows: List[Dict[str, Any]] = []

        for element in reversed_child_elements:
            element_tag_name: str = element.name
            if element_tag_name == "div":
                # Switched home_team and away_team due to reversed order
                home_team, away_team = self._parse_clubs(element=element)

                # Retrieve the fixture object that matches the provided home and away teams
                fixture: Fixture = self._retrieve_matching_fixture(
                    home_team=home_team, away_team=away_team
                )

                if not Result.objects.filter(fixture=fixture).exists():
                    # Check if the matchweek has changed (scraping only one matchweek at a time)
                    if matchweek is not None and matchweek != fixture.matchweek:
                        break

                    # Format the fixture date as DD/MM/YYYY
                    formatted_date: str = fixture.date.strftime("%d/%m/%Y")

                    # Parse home and away team scores
                    home_team_score, away_team_score = self._parse_team_scores(
                        element=element
                    )

                    # Process detailed game data to extract statistics and create row data
                    game_detailed_data: Dict[str, Any] = (
                        self._process_game_detailed_data(
                            element=element,
                        )
                    )

                    # Create row_data dictionary for the row dataframe
                    row_data: Dict[str, Any] = self._create_row_data(
                        season_end_year=self._SEASON_END_YEAR,
                        formatted_date=formatted_date,
                        home_team=home_team,
                        away_team=away_team,
                        home_team_score=home_team_score,
                        away_team_score=away_team_score,
                        game_detailed_data=game_detailed_data,
                    )

                    # Append the processed game's row data to the list
                    data_rows.append(row_data)

                    # Add result to the database and track the processed clubs
                    Result.objects.create(
                        home_team_score=home_team_score,
                        away_team_score=away_team_score,
                        home_team_goal_scorers=game_detailed_data[
                            "home_team_goal_scorers"
                        ],
                        away_team_goal_scorers=game_detailed_data[
                            "away_team_goal_scorers"
                        ],
                        home_team_possession=game_detailed_data["home_team_possession"],
                        away_team_possession=game_detailed_data["away_team_possession"],
                        home_team_shots=game_detailed_data["home_team_shots"],
                        away_team_shots=game_detailed_data["away_team_shots"],
                        home_team_shots_on_target=game_detailed_data[
                            "home_team_shots_on_target"
                        ],
                        away_team_shots_on_target=game_detailed_data[
                            "away_team_shots_on_target"
                        ],
                        home_team_shots_off_target=game_detailed_data[
                            "home_team_shots_off_target"
                        ],
                        away_team_shots_off_target=game_detailed_data[
                            "away_team_shots_off_target"
                        ],
                        home_team_shots_blocked=game_detailed_data[
                            "home_team_shots_blocked"
                        ],
                        away_team_shots_blocked=game_detailed_data[
                            "away_team_shots_blocked"
                        ],
                        home_team_passing=game_detailed_data["home_team_passing"],
                        away_team_passing=game_detailed_data["away_team_passing"],
                        home_team_clear_cut_chances=game_detailed_data[
                            "home_team_clear_cut_chances"
                        ],
                        away_team_clear_cut_chances=game_detailed_data[
                            "away_team_clear_cut_chances"
                        ],
                        home_team_corners=game_detailed_data["home_team_corners"],
                        away_team_corners=game_detailed_data["away_team_corners"],
                        home_team_offsides=game_detailed_data["home_team_offsides"],
                        away_team_offsides=game_detailed_data["away_team_offsides"],
                        home_team_tackles=game_detailed_data["home_team_tackles"],
                        away_team_tackles=game_detailed_data["away_team_tackles"],
                        home_team_aerial_duels=game_detailed_data[
                            "home_team_aerial_duels"
                        ],
                        away_team_aerial_duels=game_detailed_data[
                            "away_team_aerial_duels"
                        ],
                        home_team_fouls_committed=game_detailed_data[
                            "home_team_fouls_committed"
                        ],
                        away_team_fouls_committed=game_detailed_data[
                            "away_team_fouls_committed"
                        ],
                        home_team_fouls_won=game_detailed_data["home_team_fouls_won"],
                        away_team_fouls_won=game_detailed_data["away_team_fouls_won"],
                        home_team_yellow_cards=game_detailed_data[
                            "home_team_yellow_cards"
                        ],
                        away_team_yellow_cards=game_detailed_data[
                            "away_team_yellow_cards"
                        ],
                        home_team_red_cards=game_detailed_data["home_team_red_cards"],
                        away_team_red_cards=game_detailed_data["away_team_red_cards"],
                        fixture=fixture,
                    )

                    # Update the matchweek
                    matchweek = fixture.matchweek

        return data_rows

    @staticmethod
    def _create_table_row_from_cells(cells: ResultSet):
        """
        Parses the list of table cells and creates a TableRow object to add relevant information to the database.

        :param cells: The ResultSet representing table cells.
        :return: None
        """
        try:
            position: int = int(cells[0].text.strip())
            club_name: str = cells[1].text.strip()
            played: int = int(cells[2].text.strip())
            won: int = int(cells[3].text.strip())
            drawn: int = int(cells[4].text.strip())
            lost: int = int(cells[5].text.strip())
            goals_for: int = int(cells[6].text.strip())
            goals_against: int = int(cells[7].text.strip())
            goals_difference: int = int(cells[8].text.strip())
            points: int = int(cells[9].text.strip())
        except (ValueError, AttributeError) as e:
            # Handle conversion or attribute error
            raise Exception(f"Error parsing cell values: {e}")

        # Replace all non-alphabetic characters except spaces with a single space, then remove leading/trailing
        # spaces and extra spaces between words
        club_name = " ".join(re.sub(r"[^a-zA-Z\s]+", " ", club_name).split())

        try:
            # Find the FootballClub
            football_club = FootballClub.objects.get(name=club_name)
        except ObjectDoesNotExist:
            # Raise an exception with a custom error message
            raise Exception(f"FootballClub with name '{club_name}' does not exist")

        # Create a TableRow object and add it to the database
        TableRow.objects.create(
            position=position,
            club=football_club,
            played=played,
            won=won,
            drawn=drawn,
            lost=lost,
            goals_for=goals_for,
            goals_against=goals_against,
            goals_difference=goals_difference,
            points=points,
        )

    def process_table_html(self) -> None:
        """
        Scrapes and processes the Premier League table data from the specified URL.

        :return: None.

        :raises Exception: If the number of rows in the table is not as expected, or if fetching or parsing HTML content
        fails.
        """
        html_content: str = self._get_html_from_url(self._TABLE_URL)

        try:
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
        except Exception as e:
            raise Exception(f"Failed to parse HTML content: {e}")

        # Find all table rows
        table_rows: ResultSet = soup.find_all("tr", class_="sdc-site-table__row")

        # Expected number of rows: 20 clubs + 1 header
        expected_rows: int = 20 + 1

        if len(table_rows) != expected_rows:
            raise Exception(f"Expected {expected_rows} rows in the table.")

        # Skip the first row (header) to process only data rows
        table_rows = table_rows[1:]

        for row in table_rows:
            # Find all cells within the row
            cells: ResultSet = row.find_all("td", class_="sdc-site-table__cell")

            if len(cells) != 10:
                raise Exception(
                    f"Expected 11 cells in a data row, found {len(cells)} cells."
                )

            self._create_table_row_from_cells(cells)
