import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pyplt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import squarify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
# Set colors

colors = ["#FF69B4", "#DA70D6", "#000000", "#20B2AA", "#34495E",
          "#21618C", "#512E5F", "#45B39D", "#AAB7B8", "#16A085",
          "#FF69B4", "#00CED1", "#FF7F50", "#7FFF00", "#DA70D6"]

# Read the data frame

df = pd.read_csv("spoty.csv",encoding='latin-1')







# Shaping the dataset
print (f"Number of columns: {df.shape[1]}\nNumber of rows: {df.shape[0]}")

# Getting the Dataset info
df.info()

# Print Number of Artists
print(f"Number of artists: {df['nombre_artistas'].nunique()}\n")

# How many times do we have each artist??
artist_count = df['nombre_artistas'].value_counts()
print(artist_count)

# Fetching the df
most_used_key = df['key'].value_counts()
print(most_used_key)

# Creating the Pie Chart and center the title!
#fig = px.pie(values=most_used_key, title="Most Used Keys",names=most_used_key.index)
#fig.update_layout(title_x=0.5)
#fig.show()

## Most streamed songs chart
most_streamed_songs = df[['nombre_cancion','streams','nombre_artistas']].sort_values(by='streams', ascending=False).head(10)

#print(most_streamed_songs)

# Streams type is object hence converting it to numeric
#streams = pd.to_numeric(df['streams'])

#fig = px.bar(most_streamed_songs, title ="Top 10 streamed songs",x='nombre_cancion', y='streams', color='streams',labels={'nombre_cancion':'Song Name'},color_continuous_scale='Spectral')
#fig.show()


# Let's analyse the key mode of the top 10 streamed songs
most_streamed_songs_with_mode = df[['nombre_cancion','streams','nombre_artistas','escala']].sort_values(by='streams', ascending=False).head(10)
#print(most_streamed_songs_with_mode)

## Creating the Pie Chart and center the title!
#fig = px.pie(most_streamed_songs_with_mode, title="Most Used Mode In The 10 Streamed Songs",names='escala')
#fig.update_layout(title_x=0.5)
#fig.show()

# What about the whole data set?
mode_analysis = df['escala'].value_counts()
#print(mode_analysis)


# Let's display the pie chart again!
#fig = px.pie(values=mode_analysis, title="Most Used Keys",names=mode_analysis.index)
#fig.update_layout(title_x=0.5)
#fig.show()


# Now let's make a chart to display the most streamed songs group by key!
most_streamed_songs_by_key2 = df.groupby('key')['streams'].sum()

# Sort the data by the sum of streams in descending order
most_streamed_songs_by_key2 = most_streamed_songs_by_key2.sort_values(ascending=False)

# Format the values with a comma as the thousands separator to make it more readable in the df print
most_streamed_songs_by_key2_formatted = most_streamed_songs_by_key2.apply(lambda x: f"{x:,.0f}")

# Display the df with thousands separator
#print(most_streamed_songs_by_key2_formatted)
# Using a horizontal bar chart for this case with the original df (no formating!)
#pyplt.figure(figsize=(12, 6))
# Reading matplotlib documentation, I liked Spectral palette
#sns.barplot(x=most_streamed_songs_by_key2.values, y=most_streamed_songs_by_key2.index, palette='Spectral') 
#pyplt.xlabel('Number of Songs')
#pyplt.ylabel('Key')
#pyplt.title('Most Streamed Key')
#pyplt.show()


# Let's make a radar chart with the attributes by key.
# In this case I go for the most used key: C#!

attributes_by_key = df[['key','danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']]
attributes_by_key = attributes_by_key.query("key == 'C#'")

# Once I have queried the songs by key, I remove the key column because it is not numeric
attributes_by_CSharp = attributes_by_key.pop(attributes_by_key.columns[0])

# Calculate the average by attribute:
attributes_by_CSharp = attributes_by_key.mean()

print (attributes_by_CSharp)


# Create a radar chart with attribute names and their mean values
attributes_by_CSharp_df = pd.DataFrame({
    'attribute': ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'],
    'mean_value': attributes_by_CSharp.values
})

# Create the radar chart
import plotly.express as px
fig = px.line_polar(attributes_by_CSharp_df, r='mean_value', theta='attribute', line_close=True)
fig.update_traces(fill='toself')
#fig.show()


# Now let's make the same radar but with the attributes of the least used key: D#
# In this case I go for the most used key: C#!
attributes_by_key = df[['key','danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']]
attributes_by_key = attributes_by_key.query("key == 'D#'")

# Once I have queried the songs by key, I remove the key column because it is not numeric
attributes_by_DSharp = attributes_by_key.pop(attributes_by_key.columns[0])

# Calculate the average by attribute:
attributes_by_DSharp = attributes_by_key.mean()

print (attributes_by_DSharp)


# Create a radar chart with attribute names and their mean values
attributes_by_DSharp_df = pd.DataFrame({
    'attribute': ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'],
    'mean_value': attributes_by_DSharp.values
})

# Create the radar chart
import plotly.express as px
fig = px.line_polar(attributes_by_DSharp_df, r='mean_value', theta='attribute', line_close=True)

fig.update_traces(fill='toself')
#fig.show()

# NOW I WANT TO HAVE AN OVERLYING RADAR CHART OF THE ATTRIBUTES OF THE MOST USED KEY vs THE LEAST USED KEY: D# VS C#
# I convert my dfs into a tupple for each key 
tuple_CSharp = tuple(attributes_by_CSharp_df.itertuples(index=False, name=None))
print(tuple_CSharp)

tuple_DSharp = tuple(attributes_by_DSharp_df.itertuples(index=False, name=None))
print(tuple_DSharp)

# NOW I WANT TO HAVE AN OVERLYING RADAR CHART OF THE ATTRIBUTES OF THE MOST USED KEY vs THE LEAST USED KEY: D# VS C#
# 1. I structure my array
structured_array_CSharp = attributes_by_CSharp_df.to_records(index=False)
structured_array_DSharp = attributes_by_DSharp_df.to_records(index=False)

# 2. I convert my dfs into a tupple for each key 
tuple_CSharp = tuple(structured_array_CSharp)
tuple_DSharp = tuple(structured_array_DSharp)

# 3. Let's print the results
print(tuple_CSharp)
print(tuple_DSharp)

# Now that I have my df in tupples, I can access each of the array elements and display them.
# Fetch the values for each key so that they are passed in the "r" variable of the radar map

danceability_CSharp = tuple_CSharp[0][1]
valence_CSharp = tuple_CSharp[1][1]
energy_CSharp = tuple_CSharp[2][1]
acousticness_CSharp = tuple_CSharp[3][1]
instrumentalness_CSharp = tuple_CSharp[4][1]
liveness_CSharp = tuple_CSharp[5][1]
speechiness_CSharp = tuple_CSharp[6][1]

danceability_DSharp = tuple_DSharp[0][1]
valence_DSharp = tuple_DSharp[1][1]
energy_DSharp = tuple_DSharp[2][1]
acousticness_DSharp = tuple_DSharp[3][1]
instrumentalness_DSharp = tuple_DSharp[4][1]
liveness_DSharp = tuple_DSharp[5][1]
speechiness_DSharp = tuple_DSharp[6][1]



categories = ['Danceability','Valence','Energy',
              'Acousticness', 'Instrumentalness','Liveness','Speechines']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r= [danceability_CSharp,valence_CSharp,energy_CSharp,acousticness_CSharp,instrumentalness_CSharp,liveness_CSharp,speechiness_CSharp],
      theta=categories,
      fill='toself',
      name='C#'
))
fig.add_trace(go.Scatterpolar(
      r=[danceability_DSharp,valence_DSharp,energy_DSharp,acousticness_DSharp,instrumentalness_DSharp,liveness_DSharp,speechiness_DSharp],
      theta=categories,
      fill='toself',
      name='D#'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 80]
    )),
  showlegend=True
)

#fig.show()



## Let's make an correlation between the attributes
#pyplt.figure(figsize=(8,7))
attributes = ["danceability_%", "valence_%", "energy_%", "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%", 'bpm']
#attributes_matrix = df[attributes].corr()
#
## Create the heatmap
#sns.heatmap(attributes_matrix, annot=True,xticklabels = "auto",yticklabels = "auto",linewidths=4, cmap = "Spectral")
#pyplt.title("Attributes Correlation")
## Show the plot
#pyplt.show()
#
##We can see that the biggest correlation is between valence and danceability, followed up by energy and valence!
#pyplt.figure(figsize=(8,7))
#most_correlated= attributes_matrix[attributes_matrix>0.25]
#pyplt.title("Most Correlated Attributes")
#sns.heatmap(most_correlated, annot= True, vmin=-1, vmax=1, center=0,linewidths=4, cmap='Spectral')
##pyplt.show()
##On the other hand, the least correlated attributes are Acousticness and Energy
#pyplt.figure(figsize=(8,7))
#most_correlated= attributes_matrix[attributes_matrix<-0.25]
#pyplt.title("Least Correlated Attributes")
#sns.heatmap(most_correlated, annot= True, vmin=-1, vmax=1, center=0,linewidths=4, cmap='Spectral')
##pyplt.show()



#prom bpm
bpm_average = df['bpm'].mean()
print(bpm_average)

bpm_count = df['bpm'].value_counts()
print(bpm_count)

bpm_count_df = px.data.tips()

fig = px.density_heatmap( bpm_count_df, 
                         x=bpm_count.index, 
                       y=bpm_count.values,
                         text_auto = True,
)
#fig.show()


year = df['anio_estreno'].value_counts()

print(year)

fig = px.treemap(names=year.index, parents=['Nbr of songs/year'] * len(year), values=year.values,color_continuous_scale='Spectral')
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
#fig.show()



# I want to thank MUKUNDIYER21 for this amazing chart idea. Kudos to him!
# I did a small variation of the charts by replacing streams by BPM.

#pyplt.figure(figsize=(20,20))
#for i, feature in enumerate(attributes[:len(attributes)-1], start=1):
#    pyplt.subplot(3,3, i)
#    pyplt.title(feature.replace("_%","").capitalize() + " vs. Streams")
#    pyplt.ylabel('Streams')
#    pyplt.xlabel(feature.replace("_%","").capitalize())
#    sns.scatterplot(x=df[feature], y=df['bpm'])
#    model = LinearRegression()
#    model.fit(df[[feature]], df['bpm'])
#    y_pred = model.predict(df[[feature]])
#    sns.regplot(x=df[feature], y=y_pred, scatter=True)
#    r2 = r2_score(df['bpm'], y_pred)
#    pyplt.text(0.1, 0.9, f'R^2 = {r2:.2f}', transform=pyplt.gca().transAxes)
#    
#pyplt.show()

# Define tus datos o figuras aquí

top_artists = artist_count[:15]
artist_names = top_artists.index
counts = top_artists.values
# Datos
most_used_key = df['key'].value_counts()
# Valores para C# y D#
values_CSharp = [danceability_CSharp, valence_CSharp, energy_CSharp, acousticness_CSharp, instrumentalness_CSharp, liveness_CSharp, speechiness_CSharp]
values_DSharp = [danceability_DSharp, valence_DSharp, energy_DSharp, acousticness_DSharp, instrumentalness_DSharp, liveness_DSharp, speechiness_DSharp]
# Angulos para cada categoría
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Añade el primer valor al final para cerrar el gráfico
values_CSharp += values_CSharp[:1]
values_DSharp += values_DSharp[:1]
angles += angles[:1]


# ... Define más figuras según sea necesario

# Crea el grid de 3x3 para las figuras
plt.figure(figsize=(20, 20))

# Primera fila
plt.subplot(3, 3, 1)
plt.bar(artist_names, counts, color=colors)
# Añade etiquetas y título
plt.xlabel('Artist Names')
plt.ylabel('Count')
plt.title('Top 15 Artists with Most Songs')
# Rotación de etiquetas en el eje x para mejor legibilidad si es necesario
plt.xticks(rotation=45, ha='right')
# Descripción
plt.figtext(0.5, 0.01, 'Description 1', wrap=True, horizontalalignment='center', fontsize=8)


plt.subplot(3, 3, 2)
# Trama de barras de las 10 canciones más escuchadas
plt.bar(most_streamed_songs['nombre_cancion'], most_streamed_songs['streams'], color=colors)  # Cambio de color a verde claro
# Título y etiquetas de los ejes
plt.title('Top 10 streamed songs')
plt.xlabel('Song Name')
plt.ylabel('Streams')
# Rotar las etiquetas del eje x para mejor legibilidad si es necesario
plt.xticks(rotation=45, ha='right')
# Descripción
plt.figtext(0.5, 0.01, 'Description 3', wrap=True, horizontalalignment='center', fontsize=8)

# Mostrar la gráfica

plt.subplot(3, 3, 3)
# Gráfico de barras horizontal que muestra las canciones más transmitidas agrupadas por clave
sns.barplot(x=most_streamed_songs_by_key2.values, y=most_streamed_songs_by_key2.index, palette='Spectral')
plt.xlabel('Number of Songs')
plt.ylabel('Key')
plt.title('Most Streamed Key')
# Descripción
plt.figtext(0.5, 0.01, 'Description 6', wrap=True, horizontalalignment='center', fontsize=8)


# Segunda fila
plt.subplot(3, 3, 4)

plt.pie(most_used_key.values, labels=most_used_key.index, autopct='%1.1f%%', startangle=140)
# Añadir título centrado
plt.title('Most Used Keys', loc='center')

plt.figtext(0.5, 0.01, 'Description 2', wrap=True, horizontalalignment='center', fontsize=8)

# ... Completa la primera fila con más figuras y descripciones según sea necesario


plt.subplot(3, 3, 5)

# Gráfico de pastel que muestra el modo más utilizado en las 10 canciones más transmitidas
most_streamed_songs_with_mode = df[['nombre_cancion', 'streams', 'nombre_artistas', 'escala']].sort_values(by='streams', ascending=False).head(10)
modes_count = most_streamed_songs_with_mode['escala'].value_counts()
plt.pie(modes_count.values, labels=modes_count.index, autopct='%1.1f%%', startangle=140)
# Título centrado
plt.title('Most Used Mode In The 10 Streamed Songs', loc='center')
# Descripción
plt.figtext(0.5, 0.01, 'Description 4', wrap=True, horizontalalignment='center', fontsize=8)


# ... Completa la segunda fila con más figuras y descripciones según sea necesario



plt.subplot(3, 3, 6)


# Gráfico de radar que muestra la comparación de atributos entre C# y D#
ax = plt.subplot(3, 3, 6, polar=True)
ax.fill(angles, values_CSharp, color='blue', alpha=0.25)
ax.plot(angles, values_CSharp, color='blue', linewidth=2, linestyle='solid', label='C#')
ax.fill(angles, values_DSharp, color='red', alpha=0.25)
ax.plot(angles, values_DSharp, color='red', linewidth=2, linestyle='solid', label='D#')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.yaxis.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('Comparison of Attributes between C# and D#')
plt.figtext(0.5, 0.01, 'Description 7', wrap=True, horizontalalignment='center', fontsize=8)

# Tercera fila

plt.subplot(3, 3, 7)
# Gráfico de treemap para mostrar el recuento de canciones por año

colors = np.linspace(0, 1, len(year))
colors = np.linspace(0, 1, len(year))
squarify.plot(sizes=year.values, label=year.index, color=plt.cm.Spectral(colors), text_kwargs={'fontsize': 8, 'color': 'white', 'horizontalalignment':'left'})
plt.title('Number of Songs per Year')
plt.figtext(0.5, 0.01, 'Description 8', wrap=True, horizontalalignment='center', fontsize=8)


plt.subplot(3, 3, 8)
# Crear el mapa de calor de correlación
attributes = ["danceability_%", "valence_%", "energy_%", "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%", 'bpm']
attributes_matrix = df[attributes].corr()
sns.heatmap(attributes_matrix, annot=True, xticklabels="auto", yticklabels="auto", linewidths=4, cmap="Spectral")
plt.title("Attributes Correlation")
plt.figtext(0.5, 0.01, 'Description 9', wrap=True, horizontalalignment='center', fontsize=8)


plt.subplot(3, 3, 9)
## Scatter plot y regresión lineal para valence_
#sns.scatterplot(x=df['valence_%'], y=df['bpm'], label='Valence_')
#model_valence = LinearRegression()
#X_valence = df['valence_%'].values.reshape(-1,1)
#y_valence = df['bpm'].values
#model_valence.fit(X_valence, y_valence)
#y_pred_valence = model_valence.predict(X_valence)
#plt.plot(X_valence, y_pred_valence, color='blue', linewidth=2, label=f'Valence_ (R²={r2_score(y_valence, y_pred_valence):.2f})')  # Cambia el color a azul
#
## Scatter plot y regresión lineal para danceability
#sns.scatterplot(x=df['danceability_%'], y=df['bpm'], label='Danceability')
#model_danceability = LinearRegression()
#X_danceability = df['danceability_%'].values.reshape(-1,1)
#y_danceability = df['bpm'].values
#model_danceability.fit(X_danceability, y_danceability)
#y_pred_danceability = model_danceability.predict(X_danceability)
#plt.plot(X_danceability, y_pred_danceability, color='red', linewidth=2, label=f'Danceability (R²={r2_score(y_danceability, y_pred_danceability):.2f})')  # Cambia el color a rojo
#
## Título y etiquetas de los ejes
#plt.title("Valence_ and Danceability vs. BPM")
#plt.xlabel("Value")
#plt.ylabel("Beats per Minute")
#
## Ubicación de la R²
#plt.text(1.05, 0.5, f"Valence_ R² = {r2_score(y_valence, y_pred_valence):.2f}\nDanceability R² = {r2_score(y_danceability, y_pred_danceability):.2f}", 
#         transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')
#
## Leyenda
#plt.legend(loc='upper left')
#
#
#
#
# Scatter plot y regresión lineal para danceability
sns.scatterplot(x=df['valence_%'], y=df['danceability_%'], label='Valence_')
model_valence = LinearRegression()
X_valence = df['valence_%'].values.reshape(-1,1)
y_danceability = df['danceability_%'].values
model_valence.fit(X_valence, y_danceability)
y_pred_danceability = model_valence.predict(X_valence)
plt.plot(X_valence, y_pred_danceability, color='blue', linewidth=2, label=f'Danceability (R²={r2_score(y_danceability, y_pred_danceability):.2f})')  # Cambia el color a azul

# Scatter plot y regresión lineal para valence_
sns.scatterplot(x=df['danceability_%'], y=df['valence_%'], label='Danceability')
model_danceability = LinearRegression()
X_danceability = df['danceability_%'].values.reshape(-1,1)
y_valence = df['valence_%'].values
model_danceability.fit(X_danceability, y_valence)
y_pred_valence = model_danceability.predict(X_danceability)
plt.plot(X_danceability, y_pred_valence, color='red', linewidth=2, label=f'Valence_ (R²={r2_score(y_valence, y_pred_valence):.2f})')  # Cambia el color a rojo

# Título y etiquetas de los ejes
plt.title("Danceability and Valence_ vs. %")
plt.xlabel("Valence_ (%)")
plt.ylabel("Danceability (%)")

# Ubicación de la R²
#plt.text(0.05, 0.95, f"Valence_ R² = {r2_score(y_valence, y_pred_valence):.2f}\nDanceability R² = {r2_score(y_danceability, y_pred_danceability):.2f}", 
 #        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# Leyenda
plt.legend(loc='upper left')

# Mostrar el gráfico
#... Completa la tercera fila con más figuras y descripciones según sea necesario
plt.tight_layout()
plt.show()

