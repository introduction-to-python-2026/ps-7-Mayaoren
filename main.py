
#עושים import לכל הספריות שצריך
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()



df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].apply(lambda x: iris.target_names[x]) # הוספת שמות המינים


print("--- בדיקה ראשונית: חמש שורות ראשונות ---")
print(df.head())

print("\n--- מבנה הנתונים וסוגי הנתונים ---")
df.info()

print("\n--- תיאור סטטיסטי ---")
print(df.describe())


# מגדירים את הגרפים שיכילו את ההיסטוגרמות
df[iris.feature_names].hist(figsize=(10, 8))
plt.suptitle('לש םינייפאמה תוגלפתה Iris (תומרגוטסיה)', y=1.02)
plt.tight_layout()
plt.show()

# גרף פיזור של אורך כותרת מול רוחב כותרת, צבוע לפי מין
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='petal length (cm)',
    y='petal width (cm)',
    hue='species_name',
    data=df,
    palette='viridis',
    s=100
)
plt.title('רוזיפ ףרג: תרתוכ בחור לומ תרתוכ ךרוא (ןימ יפל עובצ)')
plt.xlabel('תרתוכ ךרוא (מ"ס)')
plt.ylabel('תרתוכ בחור (מ"ס)')
plt.legend(title='ןימ')
plt.grid(True)
plt.show()

plt.savefig('iris_correlation_scatterplot.png')

