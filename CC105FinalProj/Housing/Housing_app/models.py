from django.db import models

# Create your models here.
class HousingPrediction(models.Model):
    sqft = models.FloatField()
    bedrooms = models.IntegerField()
    bathrooms = models.FloatField()
    lot_size = models.FloatField()
    year_built = models.IntegerField()
    stories = models.IntegerField()
    neighborhood = models.CharField(max_length=50)
    garage = models.IntegerField()
    condition = models.CharField(max_length=50)
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: ${self.predicted_price:.2f} - {self.neighborhood} ({self.sqft} sqft)"
