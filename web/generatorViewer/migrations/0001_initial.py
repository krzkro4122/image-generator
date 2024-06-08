# Generated by Django 5.0.6 on 2024-06-08 23:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DatasetEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('created_at', models.DateTimeField(verbose_name='date of creation')),
                ('status', models.IntegerField(choices=[(0, 'Initial'), (1, 'Liked'), (2, 'Disliked'), (3, 'Undecided')], default=0)),
            ],
        ),
    ]
