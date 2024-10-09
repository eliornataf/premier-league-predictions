from typing import List, Tuple
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator, URLValidator
from django.utils.text import slugify


class Location(models.Model):
    """
    Represents a location.

    Fields:

        city: The city where the location is located. (CharField)
        country: The country where the location is located. (CharField)

    """

    city = models.CharField(
        max_length=30, help_text="The city where the location is located."
    )
    country = models.CharField(
        max_length=30, help_text="The country where the location is located."
    )

    class Meta:
        verbose_name = "Location"
        verbose_name_plural = "Locations"

    def __str__(self):
        return f"{self.city}, {self.country}"


class Stadium(models.Model):
    """
    Represents a football stadium.

    Fields:

        name: The name of the stadium. (CharField)
        opened: The year that the stadium was opened. (IntegerField)
        capacity: The capacity of the stadium. (IntegerField)
        interior_image: The image of the stadium's interior. (ImageField)
        exterior_image: The image of the stadium's exterior. (ImageField)
        slug: A unique identifier for the stadium. (SlugField)

    """

    name = models.CharField(max_length=30, help_text="The name of the stadium.")
    opened = models.IntegerField(
        validators=[
            MinValueValidator(1807, message="Year must be 1807 or later"),
            MaxValueValidator(2024, message="Year must be 2024 or earlier"),
        ],
        help_text="The year that the stadium was opened.",
    )
    capacity = models.IntegerField(
        validators=[
            MinValueValidator(
                5000, message="The minimum stadium capacity for PL " "clubs"
            )
        ],
        help_text="The capacity of the stadium.",
    )
    interior_image = models.ImageField(
        upload_to="stadiums/interior",
        blank=True,
        null=True,
        help_text="Image of the stadium's interior",
    )
    exterior_image = models.ImageField(
        upload_to="stadiums/exterior",
        blank=True,
        null=True,
        help_text="Image of the stadium's exterior",
    )
    slug = models.SlugField(
        max_length=30,
        unique=True,
        help_text="A unique identifier for the stadium.",
    )

    class Meta:
        verbose_name = "Stadium"
        verbose_name_plural = "Stadiums"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} (Opened: {self.opened}, Capacity: {self.capacity})"


class FootballClub(models.Model):
    """
    Represents a football club.

    Fields:

        name: The name of the football club. (CharField)
        year_founded: The year that the football club was founded. (IntegerField)
        website: The website address of the football club. (URLField)
        badge: The image of the football club's badge. (ImageField)
        slug: A unique identifier for the football club. (SlugField)
        premier_league_trophies: The number of Premier League trophies won by the club. (IntegerField)

    Relation:

        location: The location where the football club plays its home games. (ForeignKey to Location)
        stadium: The stadium where the football club plays its home games. (ForeignKey to Stadium)

    """

    name = models.CharField(max_length=30, help_text="The name of the football club.")
    year_founded = models.IntegerField(
        validators=[MinValueValidator(1865, message="Oldest active PL club")],
        help_text="The year that the football club was founded.",
    )
    website = models.URLField(
        validators=[URLValidator()],
        help_text="The website address of the football club.",
    )
    badge = models.ImageField(
        upload_to="badges", blank=True, null=True, help_text="The club's badge image."
    )
    slug = models.SlugField(
        max_length=30,
        unique=True,
        help_text="A unique identifier for the football club.",
    )
    premier_league_trophies = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of Premier League trophies won by the club.",
    )

    location = models.ForeignKey(
        to="Location",
        on_delete=models.CASCADE,
        related_name="football_clubs",
        help_text="Stores the location where the football club plays its home games.",
    )
    stadium = models.ForeignKey(
        to="Stadium",
        on_delete=models.CASCADE,
        related_name="football_clubs",
        help_text="Stores the stadium where the football club plays its home games.",
    )

    class Meta:
        verbose_name = "Football Club"
        verbose_name_plural = "Football Clubs"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Fixture(models.Model):
    """
    Represents a football game fixture.

    Fields:
        season_end_year: The year in which the season ends. (IntegerField)
        matchweek: The matchweek number in which the fixture is scheduled. (IntegerField)
        date: The date of the fixture. (DateField)
        time: The time of the fixture. (TimeField)
        predicted_outcome: The predicted outcome of the game ('H' for 'Home Team Wins', 'D' for 'Draw', 'A' for 'Away
        Team Wins'). (CharField)

    Relation:
        home_team: The home team. (ForeignKey to FootballClub)
        away_team: The away team. (ForeignKey to FootballClub)
    """

    PREDICTED_OUTCOME_CHOICES: List[Tuple[str, str]] = [
        ("H", "Home Team Wins"),
        ("D", "Draw"),
        ("A", "Away Team Wins"),
    ]

    season_end_year = models.IntegerField(
        validators=[MinValueValidator(2025)],
        help_text="The year in which the season ends.",
    )
    matchweek = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(38)],
        help_text="The matchweek number in which the fixture is scheduled.",
    )
    date = models.DateField(help_text="The date of the fixture.")
    time = models.TimeField(help_text="The time of the fixture.")
    predicted_outcome = models.CharField(
        max_length=1,
        choices=PREDICTED_OUTCOME_CHOICES,
        blank=True,
        null=True,
        help_text="The predicted outcome of the game ('H' for 'Home Team Wins', 'D' "
        "for 'Draw', 'A' for 'Away Team Wins').",
    )

    home_team = models.ForeignKey(
        to=FootballClub,
        on_delete=models.CASCADE,
        related_name="home_fixtures",
        help_text="The home team.",
    )
    away_team = models.ForeignKey(
        to=FootballClub,
        on_delete=models.CASCADE,
        related_name="away_fixtures",
        help_text="The away team.",
    )

    class Meta:
        verbose_name = "Fixture"
        verbose_name_plural = "Fixtures"

    def __str__(self):
        return f"{self.home_team} vs {self.away_team}"


class Result(models.Model):
    """
    Represents the result of a football game.

    Fields:

        slug: A unique identifier for the football club. (SlugField)
        home_team_score: The score of the home team. (IntegerField)
        away_team_score: The score of the away team. (IntegerField)
        home_team_goal_scorers: The player(s) who scored for the home team. (TextField)
        away_team_goal_scorers: The player(s) who scored for the away team. (TextField)
        home_team_possession: The home team's possession percentage. (FloatField)
        away_team_possession: The away team's possession percentage. (FloatField)
        home_team_shots: The number of shots taken by the home team. (IntegerField)
        away_team_shots: The number of shots taken by the away team. (IntegerField)
        home_team_shots_on_target: The number of shots on target taken by the home team. (IntegerField)
        away_team_shots_on_target: The number of shots on target taken by the away team. (IntegerField)
        home_team_shots_off_target: The number of shots on target taken by the home team. (IntegerField)
        away_team_shots_off_target: The number of shots on target taken by the away team. (IntegerField)
        home_team_shots_blocked: The number of blocked shots taken by the home team. (IntegerField)
        away_team_shots_blocked: The number of blocked shots taken by the away team. (IntegerField)
        home_team_passing: The percentage of passes that arrived at their destination for the home team. (FloatField)
        away_team_passing: The percentage of passes that arrived at their destination for the away team. (FloatField)
        home_team_clear_cut_chances: The number of clear-cut chances for the home team. (IntegerField)
        away_team_clear_cut_chances: The number of clear-cut chances for the away team. (IntegerField)
        home_team_corners: The number of corners won by the home team. (IntegerField)
        away_team_corners: The number of corners won by the away team. (IntegerField)
        home_team_offsides: The number of times the home team was caught offside. (IntegerField)
        away_team_offsides: The number of times the away team was caught offside. (IntegerField)
        home_team_tackles: The percentage of successful tackles for the home team. (FloatField)
        away_team_tackles: The percentage of successful tackles for the away team. (FloatField)
        home_team_aerial_duels: The percentage of successful aerial duels that the home team won. (FloatField)
        away_team_aerial_duels: The percentage of successful aerial duels that the away team won. (FloatField)
        home_team_fouls_committed: The number of fouls committed by the home team. (IntegerField)
        away_team_fouls_committed: The number of fouls committed by the away team. (IntegerField)
        home_team_fouls_won: The number of fouls won by the home team. (IntegerField)
        away_team_fouls_won: The number of fouls won by the away team. (IntegerField)
        home_team_yellow_cards: The number of yellow cards received by the home team. (IntegerField)
        away_team_yellow_cards: The number of yellow cards received by the away team. (IntegerField)
        home_team_red_cards: The number of red cards received by the home team. (IntegerField)
        away_team_red_cards: The number of red cards received by the away team. (IntegerField)

    Relation:

        fixture: The fixture associated with this result. (OneToOneField to Fixture)
    """

    slug = models.SlugField(
        max_length=70,
        unique=True,
        help_text="A unique identifier for the result.",
    )
    home_team_score = models.IntegerField(
        validators=[
            MinValueValidator(0, message="A football club game's minimum score")
        ],
        help_text="The score of the home team.",
    )
    away_team_score = models.IntegerField(
        validators=[
            MinValueValidator(0, message="A football club game's minimum score")
        ],
        help_text="The score of the away team.",
    )
    home_team_goal_scorers = models.TextField(
        help_text="The player(s) who scored for the home team."
    )
    away_team_goal_scorers = models.TextField(
        help_text="The player(s) who scored for the away team."
    )
    home_team_possession = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The home team's possession percentage.",
    )
    away_team_possession = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The away team's possession percentage.",
    )
    home_team_shots = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots taken by the home team.",
    )
    away_team_shots = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots taken by the away team.",
    )
    home_team_shots_on_target = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots on target taken by the home team.",
    )
    away_team_shots_on_target = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots on target taken by the away team.",
    )
    home_team_shots_off_target = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots off target taken by the home team.",
    )
    away_team_shots_off_target = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of shots off target taken by the away team.",
    )
    home_team_shots_blocked = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of blocked shots taken by the home team.",
    )
    away_team_shots_blocked = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of blocked shots taken by the away team.",
    )
    home_team_passing = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of passes that arrived at their destination for the home team.",
    )
    away_team_passing = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of passes that arrived at their destination for the away team.",
    )
    home_team_clear_cut_chances = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of clear-cut chances for the home team.",
    )
    away_team_clear_cut_chances = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of clear-cut chances for the away team.",
    )
    home_team_corners = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of corners won by the home team.",
    )
    away_team_corners = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of corners won by the away team.",
    )
    home_team_offsides = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of times the home team was caught offside.",
    )
    away_team_offsides = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of times the away team was caught offside.",
    )
    home_team_tackles = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of successful tackles for the home team.",
    )
    away_team_tackles = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of successful tackles for the away team.",
    )
    home_team_aerial_duels = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of successful aerial duels that the home team won.",
    )
    away_team_aerial_duels = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="The percentage of successful aerial duels that the away team won.",
    )
    home_team_fouls_committed = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of fouls committed by the home team.",
    )
    away_team_fouls_committed = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of fouls committed by the away team.",
    )
    home_team_fouls_won = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of fouls won by the home team.",
    )
    away_team_fouls_won = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of fouls won by the away team.",
    )
    home_team_yellow_cards = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of yellow cards received by the home team.",
    )
    away_team_yellow_cards = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of yellow cards received by the away team.",
    )
    home_team_red_cards = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of red cards received by the home team.",
    )
    away_team_red_cards = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The number of red cards received by the away team.",
    )

    fixture = models.OneToOneField(
        to=Fixture,
        on_delete=models.CASCADE,
        related_name="result",
        help_text="The fixture associated with this result.",
    )

    class Meta:
        verbose_name = "Result"
        verbose_name_plural = "Results"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(
                self.fixture.home_team.name + "-vs-" + self.fixture.away_team.name
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Result for {self.fixture}"


class TableRow(models.Model):
    """
    Represents a row in the Premier League table at a certain time.

    Fields:

        position: The position on the table. (IntegerField)
        played: The number of games played by the club. (IntegerField)
        won: The number of games won by the club. (IntegerField)
        drawn: The number of games drawn by the club. (IntegerField)
        lost: The number of games lost by the club. (IntegerField)
        goals_for: The goals scored by the club. (IntegerField)
        goals_against: The goals conceded by the club. (IntegerField)
        goals_difference: The difference between Goals For and Goals Against. (IntegerField)
        points: The total points earned by the club. (IntegerField)

    Relation:

        club: The football club in the table. (OneToOneField to FootballClub)
    """

    position = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(20)],
        help_text="The position on the table.",
    )
    played = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(38)],
        help_text="The number of games played by the club.",
    )
    won = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(38)],
        help_text="The number of games won by the club.",
    )
    drawn = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(38)],
        help_text="The number of games drawn by the club.",
    )
    lost = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(38)],
        help_text="The number of games lost by the club.",
    )
    goals_for = models.IntegerField(
        validators=[MinValueValidator(0)], help_text="The goals scored by the club."
    )
    goals_against = models.IntegerField(
        validators=[MinValueValidator(0)], help_text="The goals conceded by the club."
    )
    goals_difference = models.IntegerField(
        help_text="The difference between Goals For and Goals Against."
    )
    points = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="The total points earned by the club.",
    )

    club = models.OneToOneField(
        to="FootballClub",
        on_delete=models.CASCADE,
        help_text="The football club in the table.",
    )

    class Meta:
        verbose_name = "Table Row"
        verbose_name_plural = "Table Rows"

    def __str__(self):
        return f"Table Row for {self.club.name}"
