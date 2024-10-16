# Generated by Django 4.2.4 on 2023-08-19 14:08

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Fixture",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "home_team_predicted_score",
                    models.IntegerField(
                        blank=True,
                        help_text="The predicted score of the home team.",
                        null=True,
                        validators=[
                            django.core.validators.MinValueValidator(
                                0, message="A football club game's minimum score"
                            )
                        ],
                    ),
                ),
                (
                    "away_team_predicted_score",
                    models.IntegerField(
                        blank=True,
                        help_text="The predicted score of the away team.",
                        null=True,
                        validators=[
                            django.core.validators.MinValueValidator(
                                0, message="A football club game's minimum score"
                            )
                        ],
                    ),
                ),
                ("date", models.DateField(help_text="The date of the fixture.")),
                ("time", models.TimeField(help_text="The time of the fixture.")),
            ],
            options={
                "verbose_name": "Fixture",
                "verbose_name_plural": "Fixtures",
            },
        ),
        migrations.CreateModel(
            name="Location",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "city",
                    models.CharField(
                        help_text="The city where the location is located.",
                        max_length=30,
                    ),
                ),
                (
                    "country",
                    models.CharField(
                        help_text="The country where the location is located.",
                        max_length=30,
                    ),
                ),
            ],
            options={
                "verbose_name": "Location",
                "verbose_name_plural": "Locations",
            },
        ),
        migrations.CreateModel(
            name="Stadium",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        help_text="The name of the stadium.", max_length=30
                    ),
                ),
                (
                    "opened",
                    models.IntegerField(
                        help_text="The year that the stadium was opened.",
                        validators=[
                            django.core.validators.MinValueValidator(
                                1807, message="Year must be 1807 or later"
                            ),
                            django.core.validators.MaxValueValidator(
                                2023, message="Year must be 2023 or earlier"
                            ),
                        ],
                    ),
                ),
                (
                    "capacity",
                    models.IntegerField(
                        help_text="The capacity of the stadium.",
                        validators=[
                            django.core.validators.MinValueValidator(
                                5000,
                                message="The minimum stadium capacity for PL clubs",
                            )
                        ],
                    ),
                ),
            ],
            options={
                "verbose_name": "Stadium",
                "verbose_name_plural": "Stadiums",
            },
        ),
        migrations.CreateModel(
            name="Result",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "home_team_actual_score",
                    models.IntegerField(
                        help_text="The actual score of the home team.",
                        validators=[
                            django.core.validators.MinValueValidator(
                                0, message="A football club game's minimum score"
                            )
                        ],
                    ),
                ),
                (
                    "away_team_actual_score",
                    models.IntegerField(
                        help_text="The actual score of the away team.",
                        validators=[
                            django.core.validators.MinValueValidator(
                                0, message="A football club game's minimum score"
                            )
                        ],
                    ),
                ),
                (
                    "fixture",
                    models.OneToOneField(
                        help_text="The fixture associated with this result.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="result",
                        to="website.fixture",
                    ),
                ),
            ],
            options={
                "verbose_name": "Result",
                "verbose_name_plural": "Results",
            },
        ),
        migrations.CreateModel(
            name="FootballClub",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        help_text="The name of the football club.", max_length=30
                    ),
                ),
                (
                    "year_founded",
                    models.IntegerField(
                        help_text="The year that the football club was founded.",
                        validators=[
                            django.core.validators.MinValueValidator(
                                1865, message="Oldest active PL club"
                            )
                        ],
                    ),
                ),
                (
                    "website",
                    models.URLField(
                        help_text="The website address of the football club.",
                        validators=[django.core.validators.URLValidator()],
                    ),
                ),
                (
                    "badge",
                    models.ImageField(
                        blank=True,
                        help_text="The club's badge image.",
                        null=True,
                        upload_to="football_clubs",
                    ),
                ),
                (
                    "slug",
                    models.SlugField(
                        help_text="A unique identifier for the football club.",
                        max_length=30,
                        unique=True,
                    ),
                ),
                (
                    "premier_league_trophies",
                    models.IntegerField(
                        help_text="The number of Premier League trophies won by the club.",
                        validators=[django.core.validators.MinValueValidator(0)],
                    ),
                ),
                (
                    "location",
                    models.ForeignKey(
                        help_text="Stores the location where the football club plays its home games.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="football_clubs",
                        to="website.location",
                    ),
                ),
                (
                    "stadium",
                    models.ForeignKey(
                        help_text="Stores the stadium where the football club plays its home games.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="football_clubs",
                        to="website.stadium",
                    ),
                ),
            ],
            options={
                "verbose_name": "Football Club",
                "verbose_name_plural": "Football Clubs",
            },
        ),
        migrations.AddField(
            model_name="fixture",
            name="away_team",
            field=models.ForeignKey(
                help_text="The away team.",
                on_delete=django.db.models.deletion.CASCADE,
                related_name="away_fixtures",
                to="website.footballclub",
            ),
        ),
        migrations.AddField(
            model_name="fixture",
            name="home_team",
            field=models.ForeignKey(
                help_text="The home team.",
                on_delete=django.db.models.deletion.CASCADE,
                related_name="home_fixtures",
                to="website.footballclub",
            ),
        ),
    ]
