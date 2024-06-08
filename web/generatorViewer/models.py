from django.db import models

class SwipeStatus(models.IntegerChoices):
    INITIAL = 0
    LIKED = 1
    DISLIKED = 2
    UNDECIDED = 3

class DatasetEntry(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField("date of creation")
    status = models.IntegerField(
        choices=SwipeStatus.choices,
        default=SwipeStatus.INITIAL
    )

    def __str__(self):
        return f"{self.name} [{self.created_at}]"
