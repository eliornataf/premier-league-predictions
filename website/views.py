from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.core.cache.backends.base import DEFAULT_TIMEOUT
from django.views.generic import ListView, DetailView
from django.views.generic.base import TemplateView
from .models import Fixture, Result, FootballClub, TableRow, Stadium

# Get the cache timeout from Django settings, or use the default timeout if not specified
CACHE_TTL = getattr(settings, "CACHE_TTL", DEFAULT_TIMEOUT)


class StartingPageView(TemplateView):
    template_name = "website/index.html"


@method_decorator(cache_page(CACHE_TTL), name="dispatch")
class FixturesView(ListView):
    template_name = "website/fixtures.html"
    model = Fixture
    ordering = ["date", "time"]
    context_object_name = "fixtures"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        grouped_fixtures = {}
        prev_date = None
        for fixture in context["fixtures"]:
            if prev_date != fixture.date:
                grouped_fixtures[fixture.date] = []
            grouped_fixtures[fixture.date].append(fixture)
            prev_date = fixture.date
        context["grouped_fixtures"] = grouped_fixtures
        return context


@method_decorator(cache_page(CACHE_TTL), name="dispatch")
class ResultsView(ListView):
    template_name = "website/results.html"
    model = Result
    ordering = ["fixture__date", "fixture__time"]
    context_object_name = "results"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        grouped_results = {}
        prev_date = None
        for result in context["results"]:
            if prev_date != result.fixture.date:
                grouped_results[result.fixture.date] = []
            grouped_results[result.fixture.date].append(result)
            prev_date = result.fixture.date
        context["grouped_results"] = grouped_results
        return context


class ResultDetailsView(DetailView):
    template_name = "website/result_details.html"
    model = Result
    context_object_name = "result"
    slug_field = "slug"


@method_decorator(cache_page(CACHE_TTL), name="dispatch")
class TableViews(ListView):
    template_name = "website/table.html"
    model = TableRow
    ordering = ["position"]
    context_object_name = "table_rows"


@method_decorator(cache_page(CACHE_TTL), name="dispatch")
class ClubsView(ListView):
    template_name = "website/clubs.html"
    model = FootballClub
    ordering = ["name"]
    context_object_name = "football_clubs"


class ClubDetailsView(DetailView):
    template_name = "website/club_details.html"
    model = FootballClub
    context_object_name = "football_club"
    slug_field = "slug"


class StadiumDetailsView(DetailView):
    template_name = "website/stadium_details.html"
    model = Stadium
    context_object_name = "stadium"
    slug_field = "slug"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        stadium = self.get_object()
        football_club = FootballClub.objects.filter(stadium=stadium).first()
        context["football_club"] = football_club
        return context
