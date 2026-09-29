# %% [markdown]
# # Proyecto Final - Análisis y Visualización de Datos
# Este cuaderno contiene el pipeline completo de limpieza, transformación y gráficos.

# %%
# Importar librerías necesarias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# %%
# Cargar el dataset asegurando que busque en la misma carpeta del notebook
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'dataset_analisis.csv')

print(f"Cargando dataset desde: {file_path}")
df = pd.read_csv(file_path, on_bad_lines='warn')
df.head()

# %%
# 1. Limpieza de datos básica
df = df.dropna(subset=['fecha'])
df = df.dropna(subset=['categoria'])

# Rellenar nulos
df['region'] = df['region'].fillna('Desconocida')
df['descuento'] = df['descuento'].fillna(0.0)
df['vendedor'] = df['vendedor'].fillna('Desconocido')

# Borrar duplicados usando el id
df = df.drop_duplicates(subset=['id'])

# Conversión de tipos
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df['precio_unitario'] = pd.to_numeric(df['precio_unitario'], errors='coerce')
df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce')
df['descuento'] = pd.to_numeric(df['descuento'], errors='coerce').fillna(0)

# Limpieza de espacios en blanco en columnas de texto
columnas_texto = df.select_dtypes(include=['object']).columns
for col in columnas_texto:
    df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.title()

# Validar rangos lógicos
df = df[df['precio_unitario'] > 0]
df = df[df['cantidad'] > 0]
df = df[(df['descuento'] >= 0) & (df['descuento'] <= 1)]

print("Filas después de limpiar:", df.shape[0])

# %%
# 2. Transformaciones y métricas
df['venta_bruta'] = df['cantidad'] * df['precio_unitario']
df['venta_neta'] = df['venta_bruta'] * (1 - df['descuento'])

df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter

# Clasificar con pd.cut
bins = [0, 5000, 20000, 50000, np.inf]
labels = ['Baja', 'Media', 'Alta', 'Premium']
df['rango_venta'] = pd.cut(df['venta_neta'], bins=bins, labels=labels)

# Merge con metas de vendedores
metas_vendedores = pd.DataFrame({
    'vendedor': ['Ana Garcia', 'Luis Soto', 'Carlos Lopez', 'Maria Perez'],
    'meta_asignada': [500000, 300000, 250000, 200000]
})
df_limpio = pd.merge(df, metas_vendedores, on='vendedor', how='left')

df_limpio.head()

# %%
# 3. Agrupaciones y reportes de negocio
print("--- RESUMEN POR VENDEDOR ---")
resumen_vendedor = df_limpio.groupby('vendedor').agg(
    venta_total=('venta_neta', 'sum'),
    ticket_promedio=('venta_neta', 'mean'),
    total_transacciones=('id', 'count')
).round(2).sort_values('venta_total', ascending=False)
display(resumen_vendedor)

print("\n--- RESUMEN POR CATEGORIA ---")
resumen_categoria = df_limpio.groupby('categoria').agg(
    venta_neta_total=('venta_neta', 'sum'),
    descuento_promedio=('descuento', 'mean')
).round(2).sort_values('venta_neta_total', ascending=False)
display(resumen_categoria)

# %%
# 4. Generación de Gráficos (Quedarán fijos debajo de esta celda en Jupyter)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Dashboard de Resultados", fontsize=16, fontweight='bold')

formatter = ticker.FuncFormatter(lambda x, pos: f'RD$ {x:,.0f}')

# Gráfico 1: Ventas por vendedor
ventas_vend = df_limpio.groupby('vendedor')['venta_neta'].sum().reset_index()
sns.barplot(data=ventas_vend, x='vendedor', y='venta_neta', hue='vendedor', ax=axes[0, 0], palette="Blues_d", legend=False)
axes[0, 0].set_title("Ventas por Vendedor")
axes[0, 0].yaxis.set_major_formatter(formatter)

# Gráfico 2: Histograma de ventas
sns.histplot(df_limpio['venta_neta'], bins=15, kde=True, ax=axes[0, 1], color="teal")
axes[0, 1].set_title("Distribucion de Ventas")
axes[0, 1].xaxis.set_major_formatter(formatter)

# Gráfico 3: Boxplot por categoría
sns.boxplot(data=df_limpio, x='categoria', y='venta_neta', hue='categoria', ax=axes[1, 0], palette="Set2", legend=False)
axes[1, 0].set_title("Ventas por Categoria")
axes[1, 0].yaxis.set_major_formatter(formatter)
axes[1, 0].tick_params(axis='x', rotation=30)

# Gráfico 4: Línea de tiempo por mes
ventas_mes = df_limpio.groupby('mes')['venta_neta'].sum().reset_index()
sns.lineplot(data=ventas_mes, x='mes', y='venta_neta', marker="o", ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Evolucion por Mes")
axes[1, 1].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("grafico_resultado.png", dpi=300)
plt.show()