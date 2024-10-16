# Generated by Django 4.2.4 on 2023-08-23 13:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("website", "0002_alter_footballclub_badge"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="fixture",
            name="away_team_predicted_score",
        ),
        migrations.RemoveField(
            model_name="fixture",
            name="home_team_predicted_score",
        ),
        migrations.AddField(
            model_name="fixture",
            name="predicted_outcome",
            field=models.CharField(
                blank=True,
                choices=[
                    ("H", "Home Team Wins"),
                    ("D", "Draw"),
                    ("A", "Away Team Wins"),
                ],
                help_text="The predicted outcome of the game ('H' for 'Home Team Wins', 'D' for 'Draw', 'A' for 'Away Team Wins').",
                max_length=1,
                null=True,
            ),
        ),
    ]
