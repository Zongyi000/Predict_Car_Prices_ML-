from matplotlib import pyplot as plt
import pandas as pd

change_map = {'year': 13726.26798319613, 'manufacturer': 24677.832230123626, 'fuel': 10288.642079229468,
              'odometer': 7889.018027428483, 'title_status': 13236.377732408715, 'transmission': 13180.320977949652,
              'type': 11040.65588355134, 'state': 9691.783398025626}
# sort the map by value
change_map = {k: v for k, v in sorted(change_map.items(), key=lambda item: item[1])}
df_sorted = pd.DataFrame(list(change_map.items()), columns=['Feature', 'Value'])
# draw the bar chart
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['Feature'], df_sorted['Value'])
plt.xlabel('rmse after change')
plt.ylabel('Feature')
plt.title('Sensitivity Analysis')
plt.subplots_adjust(left=0.25)
plt.show()
