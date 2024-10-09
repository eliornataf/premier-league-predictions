# Generated by Django 4.2.4 on 2023-09-04 19:46

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("website", "0006_tablerow"),
    ]

    operations = [
        migrations.AddField(
            model_name="fixture",
            name="matchweek",
            field=models.IntegerField(
                default=2,
                help_text="The matchweek number in which the fixture is scheduled.",
                validators=[
                    django.core.validators.MinValueValidator(1),
                    django.core.validators.MaxValueValidator(38),
                ],
            ),
            preserve_default=False,
        ),
    ]
