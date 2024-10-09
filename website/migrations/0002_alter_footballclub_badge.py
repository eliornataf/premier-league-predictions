# Generated by Django 4.2.4 on 2023-08-21 16:21

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("website", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="footballclub",
            name="badge",
            field=models.ImageField(
                blank=True,
                help_text="The club's badge image.",
                null=True,
                upload_to="badges",
            ),
        ),
    ]
