import json
import google.generativeai as genai
from typing import Dict, Any, List
from google.generativeai import GenerativeModel
from google.generativeai.types import GenerateContentResponse
from website.models import Location, Stadium, FootballClub


class FootballClubDetailsFetcher:
    """
    A class for fetching and adding football club details to the database using language models.

    This class interacts with a language model to generate prompts and extract football club
    details. It fetches club names, generates prompts for club details, and adds the extracted
    information to the database.

    Attributes:
        _API_KEY (str): The API key for accessing the language model.

    Methods:
        add_clubs_to_database(): Fetches football club details and adds them to the database.
    """

    _API_KEY: str = "AIzaSyC9E3mh5nDqDhEkhnnq8vzFAFPS-yXChoE"

    def __init__(self):
        try:
            # Configure PaLM (Language Model) with your API key.
            genai.configure(api_key=FootballClubDetailsFetcher._API_KEY)
            # Initialize the generative model
            self._generative_model: GenerativeModel = genai.GenerativeModel(
                "gemini-pro"
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize the generative model with the provided API key."
            ) from e

    def _generate_club_list(self) -> List[str]:
        """
        Generate a list of football club names from the API response.

        :return: A list of football club names.
        """
        # Prompt to request the list of football club names
        clubs_prompt: str = (
            "Please provide me with a list of all the football clubs that play in the Premier League for the "
            "2024-2025 season, arranged from A to Z and separated by commas."
        )

        # Generate the response text using the API
        response_text: str = self._generate_text_using_api(clubs_prompt)

        # Split the response text to extract individual club names
        club_names_unprocessed: List[str] = response_text.split(",")

        # Clean and return the list of club names
        cleaned_club_names: List[str] = [
            name.strip() for name in club_names_unprocessed
        ]
        return cleaned_club_names

    @staticmethod
    def _generate_club_details_prompt(club_name: str) -> str:
        """
        Generate the prompt for club details.

        :param club_name: The name of the football club.
        :return The formatted prompt for requesting club details in JSON format.
        """
        return f""" 
                Please provide the following details for {club_name} Football Club in JSON format:
                
                1. Name (key: name [string]): Exclude "Football Club" from the name.
                2. Year Founded (key: year_founded [int]).
                3. Website URL (key: website [string]): Do not include "https://" at the beginning or "/" at the end. 
                   Add "www" at the beginning, if needed.
                4. Premier League Titles Since 1992 (key: premier_league_titles [int]): Include only titles won 
                   in the Premier League starting from the 1992-1993 season.
                5. Location (City and Country) (key: location [string], subkeys: city [string] and country [string]).
                6. Stadium (Name, Year Opened, and Capacity) (key: stadium [string], subkeys: name [string], opened 
                   [int], and capacity [int]): Provide the stadium capacity using only numerical digits, without commas.
            
                Please do not include "json" or triple quotes at the beginning and end of the string.
                """

    def _generate_text_using_api(self, prompt: str) -> str:
        """
        Generate text using the API based on the provided prompt.

        :param prompt: The prompt for generating text.
        :return: The generated text response.
        """
        try:
            response: GenerateContentResponse = self._generative_model.generate_content(
                prompt
            )
            return response.text
        except Exception as e:
            raise RuntimeError("Failed to generate text using the API.") from e

    def _generate_club_details(self, prompt: str) -> Dict[str, Any]:
        """
        Generate football club details based on the given prompt.

        :param prompt: The prompt for generating club details.
        :return: A dictionary containing the generated club details.
        """
        # Generate the response text using the API
        response_text: str = self._generate_text_using_api(prompt)

        # Find the start position of capacity
        capacity_start: int = response_text.find('"capacity":')

        # Find the comma position after capacity_start
        comma_position: int = response_text.find(",", capacity_start)
        # Note: The comma can sometimes appear within the capacity value, like 60,000

        # Check if a comma was found and proceed accordingly
        if comma_position != -1:
            # Clean the comma in the capacity value
            cleaned_data_string: str = (
                response_text[:comma_position] + response_text[comma_position + 1 :]
            )
        else:
            cleaned_data_string: str = response_text

        try:
            # Convert the cleaned JSON response to a dictionary
            return json.loads(cleaned_data_string)
        except Exception as e:
            raise RuntimeError("Failed to convert JSON response to dictionary.") from e

    def add_clubs_to_database(self):
        """
        Fetches football club details, creates related models, and adds them to the database.
        """
        # Generate a list of football club names from the API response
        club_names: List[str] = self._generate_club_list()

        # Loop through each club name and fetch details
        for club_name in club_names:
            # Generate the prompt for club details
            details_prompt: str = self._generate_club_details_prompt(club_name)

            # Generate football club details based on the prompt
            club_details: Dict[str, Any] = self._generate_club_details(details_prompt)

            # Extract stadium details
            stadium_details: Dict[str, Any] = club_details["stadium"]

            # Check if a Stadium with the same details exists in the database
            try:
                existing_stadium = Stadium.objects.filter(
                    name=stadium_details["name"],
                    opened=stadium_details["opened"],
                    capacity=stadium_details["capacity"],
                ).first()
            except Exception as e:
                raise RuntimeError("Failed to query the Stadium database.") from e

            # Handle the result of the query
            if existing_stadium:
                stadium = existing_stadium
            else:
                try:
                    stadium = Stadium.objects.create(
                        name=stadium_details["name"],
                        opened=stadium_details["opened"],
                        capacity=stadium_details["capacity"],
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create a new Stadium entry in the database."
                    ) from e

            # Extract location details
            location_details: Dict[str, Any] = club_details["location"]

            # Check if a Location with the same details exists in the database
            try:
                existing_location = Location.objects.filter(
                    city=location_details["city"], country=location_details["country"]
                ).first()
            except Exception as e:
                raise RuntimeError("Failed to query the Location database.") from e

            # Handle the result of the query
            if existing_location:
                location = existing_location
            else:
                try:
                    location = Location.objects.create(
                        city=location_details["city"],
                        country=location_details["country"],
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create a new Location entry in the database."
                    ) from e

            try:
                # Create FootballClub model instance and save to the database
                FootballClub.objects.create(
                    name=club_name,
                    year_founded=club_details["year_founded"],
                    website=club_details["website"],
                    premier_league_trophies=club_details["premier_league_trophies"],
                    location=location,
                    stadium=stadium,
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to create a new FootballClub entry in the database."
                ) from e
