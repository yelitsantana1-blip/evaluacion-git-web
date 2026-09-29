# ==========================================
# PROYECTO FINAL - ANALISIS DE DATOS
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Aqui busco la ruta donde tengo guardado este archivo y el csv
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset_analisis.csv')

print("Cargando el archivo de datos...")
df = pd.read_csv(file_path, on_bad_lines='warn')

# 1. LIMPIEZA DE DATOS
# Quito las filas que no tienen fecha ni categoria
df = df.dropna(subset=['fecha'])
df = df.dropna(subset=['categoria'])

# Relleno los espacios vacios con valores por defecto
df['region'] = df['region'].fillna('Desconocida')
df['descuento'] = df['descuento'].fillna(0.0)
df['vendedor'] = df['vendedor'].fillna('Desconocido')

# Borro los datos repetidos usando el id
df = df.drop_duplicates(subset=['id'])

# Convierto los tipos de datos para que no den error
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df['precio_unitario'] = pd.to_numeric(df['precio_unitario'], errors='coerce')
df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce')
df['descuento'] = pd.to_numeric(df['descuento'], errors='coerce').fillna(0)

# Limpiar espacios extra en los textos y poner la primera letra en mayuscula
columnas_texto = df.select_dtypes(include=['object']).columns
for col in columnas_texto:
    df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.title()

# Filtros para validar que los precios y cantidades sean reales ( mayores a 0 )
df = df[df['precio_unitario'] > 0]
df = df[df['cantidad'] > 0]
df = df[(df['descuento'] >= 0) & (df['descuento'] <= 1)]

# 2. TRANSFORMACIONES
# Calculo la venta bruta y la neta restando el descuento
df['venta_bruta'] = df['cantidad'] * df['precio_unitario']
df['venta_neta'] = df['venta_bruta'] * (1 - df['descuento'])

# Saco el mes y el trimestre de la fecha
df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter

# Dividir las ventas por rangos con pd.cut
bins = [0, 5000, 20000, 50000, np.inf]
labels = ['Baja', 'Media', 'Alta', 'Premium']
df['rango_venta'] = pd.cut(df['venta_neta'], bins=bins, labels=labels)

# Unir con la tablita de metas de los vendedores
metas_vendedores = pd.DataFrame({
    'vendedor': ['Ana Garcia', 'Luis Soto', 'Carlos Lopez', 'Maria Perez'],
    'meta_asignada': [500000, 300000, 250000, 200000]
})
df_limpio = pd.merge(df, metas_vendedores, on='vendedor', how='left')

print("Datos limpios listos. Total de filas:", df_limpio.shape[0])

# ==========================================
# 3. AGRUPACIONES Y REPORTES
# ==========================================
print("\n--- RESUMEN POR VENDEDOR ---")
resumen_vendedor = df_limpio.groupby('vendedor').agg(
    venta_total=('venta_neta', 'sum'),
    ticket_promedio=('venta_neta', 'mean'),
    total_transacciones=('id', 'count')
).round(2).sort_values('venta_total', ascending=False)
print(resumen_vendedor.to_string())

print("\n--- RESUMEN POR CATEGORIA ---")
resumen_categoria = df_limpio.groupby('categoria').agg(
    venta_neta_total=('venta_neta', 'sum'),
    descuento_promedio=('descuento', 'mean')
).round(2).sort_values('venta_neta_total', ascending=False)
print(resumen_categoria.to_string())

print("\n--- VENTAS POR TRIMESTRE ---")
resumen_trimestre = df_limpio.groupby('trimestre').agg(
    ventas_trimestre=('venta_neta', 'sum'),
    transacciones=('id', 'count')
).round(2)
print(resumen_trimestre.to_string())

# ==========================================
# 4. GRAFICOS
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Graficos del Proyecto Final", fontsize=16, fontweight='bold')

formatter = ticker.FuncFormatter(lambda x, pos: f'RD$ {x:,.0f}')

# Grafico 1: Ventas por vendedor
ventas_vend = df_limpio.groupby('vendedor')['venta_neta'].sum().reset_index()
sns.barplot(data=ventas_vend, x='vendedor', y='venta_neta', hue='vendedor', ax=axes[0, 0], palette="Blues_d", legend=False)
axes[0, 0].set_title("Ventas por Vendedor")
axes[0, 0].yaxis.set_major_formatter(formatter)

# Grafico 2: Histograma de ventas
sns.histplot(df_limpio['venta_neta'], bins=15, kde=True, ax=axes[0, 1], color="teal")
axes[0, 1].set_title("Distribucion de Ventas")
axes[0, 1].xaxis.set_major_formatter(formatter)

# Grafico 3: Boxplot por categoria
sns.boxplot(data=df_limpio, x='categoria', y='venta_neta', hue='categoria', ax=axes[1, 0], palette="Set2", legend=False)
axes[1, 0].set_title("Ventas por Categoria")
axes[1, 0].yaxis.set_major_formatter(formatter)
axes[1, 0].tick_params(axis='x', rotation=30)

# Grafico 4: Linea de tiempo por mes
ventas_mes = df_limpio.groupby('mes')['venta_neta'].sum().reset_index()
sns.lineplot(data=ventas_mes, x='mes', y='venta_neta', marker="o", ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Evolucion por Mes")
axes[1, 1].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("grafico_resultado.png", dpi=300)
plt.show()

print("\n¡Listo! Todo corrio bien y se guardo la imagen.")