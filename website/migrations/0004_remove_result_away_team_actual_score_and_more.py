# Generated by Django 4.2.4 on 2023-08-29 12:02

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("website", "0003_remove_fixture_away_team_predicted_score_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="result",
            name="away_team_actual_score",
        ),
        migrations.RemoveField(
            model_name="result",
            name="home_team_actual_score",
        ),
        migrations.AddField(
            model_name="result",
            name="away_team_score",
            field=models.IntegerField(
                help_text="The score of the away team.",
                null=True,
                validators=[
                    django.core.validators.MinValueValidator(
                        0, message="A football club game's minimum score"
                    )
                ],
            ),
        ),
        migrations.AddField(
            model_name="result",
            name="home_team_score",
            field=models.IntegerField(
                help_text="The score of the home team.",
                null=True,
                validators=[
                    django.core.validators.MinValueValidator(
                        0, message="A football club game's minimum score"
                    )
                ],
            ),
        ),
    ]
