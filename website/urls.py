from functools import partial
from typing import List
from django.urls import path
from . import views

urlpatterns: List[partial] = [
    path(route="", view=views.StartingPageView.as_view(), name="starting-page"),
    path(route="fixtures", view=views.FixturesView.as_view(), name="fixtures-page"),
    path(route="results", view=views.ResultsView.as_view(), name="results-page"),
    path(
        route="results/<slug:slug>",
        view=views.ResultDetailsView.as_view(),
        name="result-detail-page",
    ),
    path(route="table", view=views.TableViews.as_view(), name="table-page"),
    path(route="clubs", view=views.ClubsView.as_view(), name="clubs-page"),
    path(
        route="clubs/<slug:slug>",
        view=views.ClubDetailsView.as_view(),
        name="club-detail-page",
    ),
    path(
        route="stadiums/<slug:slug>",
        view=views.StadiumDetailsView.as_view(),
        name="stadium-detail-page",
    ),
]
