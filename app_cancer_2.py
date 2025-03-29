import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_breast_cancer

# Configuración
st.set_page_config(page_title="Clasificador de Cáncer de Mama", layout="wide")
st.title("🧬 Clasificador de Cáncer de Mama")
shap.initjs()

# Cargar modelo y dataset
modelo = joblib.load("modelo_cancer.pkl")
data = load_breast_cancer()
features = data.feature_names
X = pd.DataFrame(data.data, columns=features)
y = pd.Series(data.target)
df = X.copy()
df["diagnóstico"] = y.map({0: "Maligno", 1: "Benigno"})

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧮 Predicción",
    "📊 Exploración",
    "🔍 SHAP",
    "📈 Informe del paciente",
    "🕘 Historial",
    "🧪 Simulador simple",
    "🧪 Comparación de escenarios"
])



# --------------------------
# TAB 1: Predicción
# --------------------------
with tab1:
    st.header("🧮 Clasificación")

    inputs = []
    cols = st.columns(3)
    for i in range(len(features)):
        col = cols[i % 3]
        with col:
            min_val = float(np.min(data.data[:, i]))
            max_val = float(np.max(data.data[:, i]))
            mean_val = float(np.mean(data.data[:, i]))
            if min_val == max_val:
                max_val += 1.0
            val = st.slider(
                label=features[i],
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.01
            )
            inputs.append(val)

    if st.button("🔍 Predecir"):
        try:
            pred = modelo.predict([inputs])[0]
            st.success("✅ Resultado: **Benigno**" if pred == 1 else "⚠️ Resultado: **Maligno**")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

# --------------------------
# TAB 2: Exploración
# --------------------------
with tab2:
    st.header("📊 Análisis y visualización")

    # Filtro por diagnóstico
    st.subheader("🔎 Filtro del dataset")
    opcion_clase = st.selectbox("Filtrar por diagnóstico:", ["Todos", "Benigno", "Maligno"])
    if opcion_clase != "Todos":
        df = df[df["diagnóstico"] == opcion_clase]
    st.dataframe(df.head(10))

    # Selector de variable
    st.subheader("📈 Variable individual")
    feature_seleccionada = st.selectbox("Seleccioná una variable para graficar:", features)

    # Histograma interactivo con línea del paciente
    with st.expander("📊 Distribución (Plotly)"):
        fig = px.histogram(
            df,
            x=feature_seleccionada,
            nbins=30,
            color_discrete_sequence=["#7FDBFF"]
        )
        if 'inputs' in locals():
            idx = list(features).index(feature_seleccionada)
            fig.add_vline(
                x=inputs[idx],
                line_dash="dash",
                line_color="crimson",
                annotation_text="Valor actual",
                annotation_position="top right"
            )
        fig.update_layout(
            height=400,
            title=feature_seleccionada,
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True, key="1")

    # Heatmap de correlaciones
    with st.expander("📌 Mapa de calor de correlaciones"):
        corr = df[features].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig.update_layout(
            title="Correlaciones entre variables",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="2" )

# --------------------------
# TAB 3: Explicación con SHAP
# --------------------------

with tab3:
    st.header("🤖 Explicabilidad del modelo")

    try:
        input_array = np.array(inputs).reshape(1, -1)

        # Crear el TreeExplainer
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(input_array)

        # Extraer valores de la clase 1 (benigno)
        shap_vals = shap_values[1][0]
        feature_impact = pd.DataFrame({
            "feature": features,
            "shap_value": shap_vals
        })

        # Ordenar por impacto absoluto
        feature_impact["abs"] = np.abs(feature_impact["shap_value"])
        feature_impact = feature_impact.sort_values("abs", ascending=True).tail(10)

        # Crear gráfico con Plotly
        fig = px.bar(
            feature_impact,
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title="Top 10 variables más influyentes",
            labels={"shap_value": "Impacto SHAP", "feature": "Variable"}
        )

        fig.update_layout(
            height=700,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="",
            xaxis_title="",
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True, key="3")

    except Exception as e:
        st.error(f"⚠️ No se pudo generar la explicación SHAP: {e}")


with tab4:
    st.header("📈 Informe del paciente")

    try:
        input_array = np.array(inputs).reshape(1, -1)

        # Predicción y probabilidad
        pred = modelo.predict(input_array)[0]
        proba = modelo.predict_proba(input_array)[0][1]

        # SHAP con TreeExplainer
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(input_array)
        shap_vals = shap_values[1][0]

        # Top 10 variables
        feature_impact = pd.DataFrame({
            "feature": features,
            "shap_value": shap_vals
        })
        feature_impact["abs"] = np.abs(feature_impact["shap_value"])
        feature_impact = feature_impact.sort_values("abs", ascending=True).tail(10)

        # Gráfico Plotly
        fig_shap = px.bar(
            feature_impact,
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title="Top 10 variables más influyentes",
            labels={"shap_value": "Impacto SHAP", "feature": "Variable"}
        )
        fig_shap.update_layout(
            height=700,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="",
            xaxis_title="",
            coloraxis_showscale=False
        )

        # Layout en columnas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🩺 Predicción", "Benigno" if pred == 1 else "Maligno")
            st.metric("🔢 Probabilidad (Benigno)", f"{proba:.2%}")
        with col2:
            st.plotly_chart(fig_shap, use_container_width=True, key="4")

    except Exception as e:
        st.error(f"⚠️ No se pudo generar el informe del paciente: {e}")

    # ✅ Esto va FUERA del try, pero dentro de tab4 y bien indentado
    st.subheader("📤 Compartir resumen del caso")

    resumen = f"📄 Resumen del caso:\n"
    resumen += f"─────────────────────────────\n"
    resumen += f"🩺 Diagnóstico: {'Benigno' if pred == 1 else 'Maligno'}\n"
    resumen += f"📊 Probabilidad: {proba:.2%}\n"
    resumen += f"🕐 Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    resumen += "📈 Valores de las variables:\n"

    for name, value in zip(features, inputs):
        resumen += f"• {name}: {round(value, 3)}\n"

    st.code(resumen.strip())

    import streamlit.components.v1 as components

    components.html(f"""
    <textarea id="texto-a-copiar" style="display:none">{resumen.strip()}</textarea>
    <button onclick="navigator.clipboard.writeText(document.getElementById('texto-a-copiar').value)"
            style="background-color:#4CAF50; color:white; padding:8px 12px; border:none; border-radius:5px; cursor:pointer;">
        📋 Copiar al portapapeles
    </button>
""", height=50)


    # Botón para descargar resumen como .txt
    st.download_button(
        label="📥 Descargar resumen como archivo .txt",
        data=resumen,
        file_name="resumen_caso.txt",
        mime="text/plain"
    )

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Generar PDF en memoria
pdf_buffer = io.BytesIO()
c = canvas.Canvas(pdf_buffer, pagesize=letter)
textobject = c.beginText(40, 750)  # Margen izquierdo y alto inicial

# Escribimos cada línea del resumen
for line in resumen.strip().split("\n"):
    textobject.textLine(line)

c.drawText(textobject)
c.showPage()
c.save()

# Preparar para descarga
pdf_buffer.seek(0)

st.download_button(
    label="🧾 Descargar informe en PDF",
    data=pdf_buffer,
    file_name="informe_caso.pdf",
    mime="application/pdf"
)


with tab5:
    st.header("🕘 Historial de predicciones")

    if "historial" not in st.session_state:
        st.session_state.historial = []

    # Solo se guarda si hay predicción actual disponible
    try:
        input_array = np.array(inputs).reshape(1, -1)
        pred = modelo.predict(input_array)[0]
        proba = modelo.predict_proba(input_array)[0][1]
        
        if st.button("💾 Guardar este caso"):
            st.session_state.historial.append({
                "Diagnóstico": "Benigno" if pred == 1 else "Maligno",
                "Probabilidad Benigno": round(proba, 4),
                "Fecha y hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("✅ Caso guardado en el historial")

    except Exception as e:
        st.info("Realizá una predicción primero para guardar el caso.")

    # Mostrar historial
    if st.session_state.historial:
        st.dataframe(pd.DataFrame(st.session_state.historial))

        # Botón para limpiar
        if st.button("🧹 Limpiar historial"):
            st.session_state.historial = []
            st.experimental_rerun()
    else:
        st.info("No hay casos guardados aún.")

with tab6:
    st.header("🧪 Simulador de impacto de variable")

    # Elegir variable a modificar
    variable_simulada = st.selectbox("Seleccioná una variable para modificar:", features)

    # Buscar índice y valores
    i = list(features).index(variable_simulada)
    original_value = inputs[i]
    min_val = float(np.min(data.data[:, i]))
    max_val = float(np.max(data.data[:, i]))

    nuevo_valor = st.slider(
        f"Modificar valor de '{variable_simulada}'",
        min_value=min_val,
        max_value=max_val,
        value=original_value,
        step=0.1
    )

    # Crear nueva entrada con el valor modificado
    inputs_sim = inputs.copy()
    inputs_sim[i] = nuevo_valor

    # Nueva predicción
    pred_sim = modelo.predict([inputs_sim])[0]
    proba_sim = modelo.predict_proba([inputs_sim])[0][1]

    # Mostrar comparación
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicción original", "Benigno" if pred == 1 else "Maligno")
        st.metric("Probabilidad original", f"{proba:.2%}")
    with col2:
        st.metric("Con valor simulado", "Benigno" if pred_sim == 1 else "Maligno")
        st.metric("Probabilidad simulada", f"{proba_sim:.2%}")

with tab7:
    st.header("🧪 Comparador de escenarios (Top 5 variables)")

    # Obtener las 5 variables más importantes del modelo
    importancias = modelo.feature_importances_
    df_importancia = pd.DataFrame({"feature": features, "importance": importancias})
    top_features = df_importancia.sort_values("importance", ascending=False)["feature"].head(5).tolist()

    # Mostrar sliders para las top 5
    st.markdown("### 🅰️ Escenario A (original)")
    inputs_a_mod = []
    cols_a = st.columns(2)
    for i, feature in enumerate(top_features):
        idx = list(features).index(feature)
        with cols_a[i % 2]:
            val = st.slider(
                f"{feature} (A)",
                min_value=float(np.min(data.data[:, idx])),
                max_value=float(np.max(data.data[:, idx])),
                value=float(inputs[idx]),
                step=0.1
            )
            inputs_a_mod.append((idx, val))

    inputs_a = inputs.copy()
    for idx, val in inputs_a_mod:
        inputs_a[idx] = val

    st.markdown("---")

    st.markdown("### 🅱️ Escenario B (modificado)")
    inputs_b_mod = []
    cols_b = st.columns(2)
    for i, feature in enumerate(top_features):
        idx = list(features).index(feature)
        with cols_b[i % 2]:
            val = st.slider(
                f"{feature} (B)",
                min_value=float(np.min(data.data[:, idx])),
                max_value=float(np.max(data.data[:, idx])),
                value=float(inputs[idx]),
                step=0.1
            )
            inputs_b_mod.append((idx, val))

    inputs_b = inputs.copy()
    for idx, val in inputs_b_mod:
        inputs_b[idx] = val

    # Predicciones
    pred_a = modelo.predict([inputs_a])[0]
    proba_a = modelo.predict_proba([inputs_a])[0][1]

    pred_b = modelo.predict([inputs_b])[0]
    proba_b = modelo.predict_proba([inputs_b])[0][1]

    # Resultados comparativos
    st.markdown("### 📊 Comparación de resultados")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicción A", "Benigno" if pred_a == 1 else "Maligno")
        st.metric("Probabilidad A", f"{proba_a:.2%}")

    with col2:
        st.metric("Predicción B", "Benigno" if pred_b == 1 else "Maligno")
        st.metric("Probabilidad B", f"{proba_b:.2%}")

