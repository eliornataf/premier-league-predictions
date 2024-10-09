# Generated by Django 4.2.4 on 2023-09-13 19:27

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("website", "0015_remove_stadium_image_stadium_exterior_image_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="stadium",
            name="slug",
            field=models.SlugField(
                blank=True,
                help_text="A unique identifier for the stadium.",
                max_length=30,
                null=True,
                unique=True,
            ),
        ),
    ]
