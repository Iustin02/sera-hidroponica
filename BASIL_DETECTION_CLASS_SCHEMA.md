# Basil Detection Class Schema

Final 3-class schema for the hydroponic control prototype.

## Classes

- `healthy` (planta arată normal)
- `nutrient_issue` (deficit sau dezechilibru nutritiv)
- `disease_issue` (simptome de boală sau infecție)

## Control mapping

- `healthy` -> hold / no action
- `nutrient_issue` -> adjust EC / pH / nutrient dosing
- `disease_issue` -> alert / ventilation / isolation

## Notes

- Keep the names stable for code and labels.
- Keep the short Romanian explanation in parentheses for readability.
- This schema is lightweight and easier to train/deploy on Raspberry Pi 5.
