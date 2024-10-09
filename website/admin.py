from django.contrib import admin
from .models import Location, Stadium, FootballClub, Fixture, Result, TableRow


class LocationAdmin(admin.ModelAdmin):
    list_filter = ("city", "country")
    list_display = ("city", "country")
    ordering = ("city",)


class StadiumAdmin(admin.ModelAdmin):
    list_display = ("name", "opened", "capacity")
    ordering = ("name",)


class FootballClubAdmin(admin.ModelAdmin):
    list_filter = ("name",)
    list_display = ("name",)
    ordering = ("name",)


class FixtureAdmin(admin.ModelAdmin):
    list_filter = ("matchweek", "date", "home_team", "away_team")
    list_display = (
        "matchweek",
        "date",
        "time",
        "home_team",
        "away_team",
        "predicted_outcome",
    )
    ordering = ["date", "time"]


class ResultAdmin(admin.ModelAdmin):
    list_filter = (
        "fixture__matchweek",
        "fixture__home_team__name",
        "fixture__away_team__name",
        "fixture__date",
    )
    list_display = (
        "fixture_matchweek",
        "fixture_home_team_name",
        "fixture_away_team_name",
        "fixture_date",
        "home_team_score",
        "away_team_score",
    )
    ordering = ["fixture__date", "fixture__time"]

    def fixture_matchweek(self, obj):
        return obj.fixture.matchweek

    fixture_matchweek.short_description = "Fixture Matchweek"

    def fixture_home_team_name(self, obj):
        return obj.fixture.home_team.name

    fixture_home_team_name.short_description = "Home Team Name"

    def fixture_away_team_name(self, obj):
        return obj.fixture.away_team.name

    fixture_away_team_name.short_description = "Away Team Name"

    def fixture_date(self, obj):
        return obj.fixture.date

    fixture_date.short_description = "Fixture Date"


class TableRowAdmin(admin.ModelAdmin):
    list_filter = ("position", "club__name")
    list_display = ("position", "club_name", "points")
    ordering = ("position",)

    def club_name(self, obj):
        return obj.club.name

    club_name.short_description = "Club Name"


admin.site.register(Location, LocationAdmin)
admin.site.register(Stadium, StadiumAdmin)
admin.site.register(FootballClub, FootballClubAdmin)
admin.site.register(Fixture, FixtureAdmin)
admin.site.register(Result, ResultAdmin)
admin.site.register(TableRow, TableRowAdmin)
