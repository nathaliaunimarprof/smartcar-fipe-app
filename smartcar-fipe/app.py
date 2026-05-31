"""
Dashboard de Predição de Preços de Veículos FIPE

Uma aplicação Streamlit para predizer preços de veículos com base em dados da FIPE.
Utiliza um modelo Random Forest treinado com R² de 93,7%.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import date

# Configuração da página
st.set_page_config(
    page_title="Preditor de Preços FIPE",
    page_icon=":material/directions_car:",
    layout="wide",
)

st.markdown("""
    <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #07111F !important;
        }

        .main-header {
            text-align: center;
            padding: 2rem 0;
            background-color: #0D3347;
            border-bottom: 1px solid #0F6E56;
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: #ffffff;
            margin-bottom: 0.25rem;
        }
        .main-header p {
            color: #5DCAA5;
            margin: 0;
        }

        .metric-card {
            background-color: #0B1F33;
            border: 0.5px solid #0F6E56;
            padding: 1rem;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #5DCAA5;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 500;
            color: #5DCAA5;
        }
        .metric-value.neutral {
            color: #ffffff;
        }

        [data-testid="stSelectbox"] > div,
        [data-testid="stTextInput"] > div > div {
            background-color: #0D3347 !important;
            border: 0.5px solid #0F6E56 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background-color: #1D9E75 !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] div[class*="track"] {
            background-color: #0F6E56 !important;
        }

        label, .stLabel, [data-testid="stWidgetLabel"] p {
            color: #5DCAA5 !important;
            font-size: 0.8rem !important;
        }

        .info-box {
            background-color: #0D3347;
            border: 0.5px solid #0F6E56;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            color: #5DCAA5;
            font-size: 0.85rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid #0F6E56 !important;
            gap: 0 !important;
            display: flex !important;
        }
        [data-testid="stTabs"] button[role="tab"] {
            color: #5DCAA5 !important;
            background: transparent !important;
            padding: 0.5rem 1.5rem !important;
            margin-right: 4px !important;
            white-space: nowrap !important;
            min-width: fit-content !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 2px solid #1D9E75 !important;
            color: #5DCAA5 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #0B1F33 !important;
            border-right: 0.5px solid #0F6E56 !important;
        }

        [data-testid="stButton"] button[kind="primary"] {
            background-color: #1D9E75 !important;
            border: none !important;
            color: #ffffff !important;
        }
        [data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #0F6E56 !important;
        }

        p, span, div {
            color: #e2e8f0;
        }
    </style>
""", unsafe_allow_html=True)


# Carrega o modelo 
@st.cache_resource
def load_model_artifacts():
    """Carrega o modelo treinado, os encoders e os metadados."""
    model_dir = Path("model")

    try:
        model = joblib.load(model_dir / "melhor_modelo.pkl")
        encoders = joblib.load(model_dir / "label_encoders.pkl")
        features = joblib.load(model_dir / "features.pkl")

        with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, encoders, features, metadata
    except Exception as e:
        st.error(f"Erro ao carregar os artefatos do modelo: {e}")
        return None, None, None, None


# Inicializa o estado da sessão
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# Carrega o modelo
model, encoders, features, metadata = load_model_artifacts()

if model is None:
    st.error("Falha ao carregar o modelo. Verifique se os arquivos existem no diretório 'model'.")
    st.stop()

# Cabeçalho
st.markdown("""
    <div class="main-header">
        <h1>SmartCar FIPE - Preditor de Preços de Veículos</h1>
        <p>Previsão de preços de veículos com Machine Learning e dados FIPE</p>
    </div>
""", unsafe_allow_html=True)

# Seção de informações do modelo
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modelo", metadata["model_name"], border=True)
with col2:
    st.metric("R²", f"{metadata['metrics_test']['r2']:.4f}", border=True)
with col3:
    st.metric("MAE", f"R$ {metadata['metrics_test']['mae']:,.0f}", border=True)
with col4:
    st.metric("MAPE", f"{metadata['metrics_test']['mape']:.2f}%", border=True)

st.caption(f"Amostras de treinamento: {metadata['train_samples']:,} | Criado em: {metadata['created_at'][:10]}")

# Layout principal
tab1, tab2, tab3 = st.tabs(["Predição de Preço", "Insights do Modelo", "Histórico de Predições"])

# Aba 1: Predição de preço
with tab1:
    st.header("Estimar o preço do veículo")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Informações do Veículo")

        brand_options = sorted(encoders["brand"].classes_)
        selected_brand = st.selectbox(
            "Marca",
            brand_options,
            index=brand_options.index("VW - VolksWagen") if "VW - VolksWagen" in brand_options else 0,
        )

        current_year = date.today().year
        year_model = st.slider(
            "Ano do Modelo",
            min_value=1980,
            max_value=current_year + 1,
            value=2020,
            step=1,
        )

        engine_size = st.selectbox(
            "Tamanho do Motor (L)",
            [1.0, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            index=2,
        )

        fuel_options = sorted(encoders["fuel"].classes_)
        selected_fuel = st.selectbox("Tipo de Combustível", fuel_options)

    with col_right:
        st.subheader("Detalhes Adicionais")

        gear_options = sorted(encoders["gear"].classes_)
        selected_gear = st.selectbox("Câmbio", gear_options)

        col_y, col_m = st.columns(2)
        with col_y:
            ref_years = list(range(2021, current_year + 1))
            year_of_reference = st.selectbox(
                "Ano de Referência",
                ref_years,
                index=len(ref_years) - 1,
            )
        with col_m:
            month_options = sorted(encoders["month_of_reference"].classes_)
            month_of_reference = st.selectbox("Mês de Referência", month_options)

        car_age = year_of_reference - year_model
        is_luxury = 1 if selected_brand in ["BMW", "Mercedes-Benz", "Audi", "Porsche", "Land Rover"] else 0

        st.info(f"Idade do veículo: {car_age} anos | Luxo: {'Sim' if is_luxury else 'Não'}")

    # Botão de predição
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        if st.button("Predizer Preço", type="primary", use_container_width=True):

            # Validação dentro do bloco do botão 
            if car_age < 0:
                st.error("O ano do modelo não pode ser maior que o ano de referência.")
            else:
                input_data = {
                    "year_of_reference": year_of_reference,
                    "month_of_reference_enc": encoders["month_of_reference"].transform([month_of_reference])[0],
                    "engine_size": engine_size,
                    "year_model": year_model,
                    "car_age": car_age,
                    "is_luxury": is_luxury,
                    "brand_enc": encoders["brand"].transform([selected_brand])[0],
                    "fuel_enc": encoders["fuel"].transform([selected_fuel])[0],
                    "gear_enc": encoders["gear"].transform([selected_gear])[0],
                }

                X_input = pd.DataFrame([input_data])[features]

                pred_log = model.predict(X_input)[0]
                pred_price = np.expm1(pred_log)

                # Intervalo de confiança usando RMSE
                rmse = metadata['metrics_test']['rmse']
                lower_bound = max(0, pred_price - 1.96 * rmse)
                upper_bound = pred_price + 1.96 * rmse

                prediction_record = {
                    "Marca": selected_brand,
                    "Year": year_model,
                    "Fuel": selected_fuel,
                    "Gear": selected_gear,
                    "Engine": f"{engine_size}L",
                    "Predicted Price": pred_price,
                    "Lower Bound": lower_bound,
                    "Upper Bound": upper_bound,
                }
                st.session_state.predictions_history.append(prediction_record)

                st.success("Preço predito com sucesso!")

                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                with result_col2:
                    with st.container(border=True):
                        st.markdown("### Preço Estimado")
                        st.metric(
                            "",
                            f"R$ {pred_price:,.2f}",
                            delta=f"±{metadata['metrics_test']['mape']:.1f}% de erro percentual médio",
                        )

                st.caption(f"Intervalo de confiança de 95%: R$ {lower_bound:,.2f} — R$ {upper_bound:,.2f}")

# Aba 2: Insights do modelo
with tab2:
    st.header("Informações do Modelo")

    if hasattr(model, "feature_importances_"):
        st.subheader("Importância das Variáveis")

        importances = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=False)

        importance_df = pd.DataFrame({
            "Variável": importances.index,
            "Importância": importances.values
        })
        importance_df["Variável"] = importance_df["Variável"].str.replace("_", " ").str.title()

        st.dataframe(
            importance_df,
            column_config={
                "Variável": st.column_config.TextColumn("Variável"),
                "Importância": st.column_config.ProgressColumn(
                    "Importância",
                    min_value=0,
                    max_value=importance_df["Importância"].max(),
                    format="%.4f",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        with st.container(border=True):
            st.subheader("Métricas de Desempenho")
            st.metric("R²", f"{metadata['metrics_test']['r2']:.4f}")
            st.metric("MAE", f"R$ {metadata['metrics_test']['mae']:,.2f}")
            st.caption("Quanto maior o R², melhor. Quanto menor o MAE, melhor.")

    with metrics_col2:
        with st.container(border=True):
            st.subheader("Métricas de Erro")
            st.metric("RMSE", f"R$ {metadata['metrics_test']['rmse']:,.2f}")
            st.metric("MAPE", f"{metadata['metrics_test']['mape']:.2f}%")
            st.caption("Erro percentual médio nas predições.")

    with st.expander("Descrição das Variáveis"):
        feature_descriptions = {
            "year_of_reference": "Ano de referência da tabela FIPE",
            "month_of_reference_enc": "Mês de referência codificado",
            "engine_size": "Cilindrada do motor em litros",
            "year_model": "Ano de fabricação do veículo",
            "car_age": "Idade calculada do veículo",
            "is_luxury": "Indicador para marcas premium (BMW, Mercedes etc.)",
            "brand_enc": "Marca do veículo codificada",
            "fuel_enc": "Tipo de combustível codificado",
            "gear_enc": "Tipo de câmbio codificado",
        }

        desc_df = pd.DataFrame([
            {"Variável": feat, "Descrição": desc}
            for feat, desc in feature_descriptions.items()
        ])
        st.dataframe(desc_df, hide_index=True, use_container_width=True)

# Aba 3: Histórico de predições
with tab3:
    st.header("Predições Recentes")

    if st.session_state.predictions_history:
        history_df = pd.DataFrame(st.session_state.predictions_history)

        st.dataframe(
            history_df,
            column_config={
                "Marca": st.column_config.TextColumn("Marca"),
                "Year": st.column_config.NumberColumn("Ano"),
                "Fuel": st.column_config.TextColumn("Combustível"),
                "Gear": st.column_config.TextColumn("Câmbio"),
                "Engine": st.column_config.TextColumn("Motor"),
                "Predicted Price": st.column_config.NumberColumn(
                    "Preço",
                    format="R$ %.2f",
                ),
                "Lower Bound": st.column_config.NumberColumn(
                    "Mín. (95%)",
                    format="R$ %.2f",
                ),
                "Upper Bound": st.column_config.NumberColumn(
                    "Máx. (95%)",
                    format="R$ %.2f",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Download do histórico como CSV
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar histórico como CSV",
            data=csv_data,
            file_name="historico_predicoes_fipe.csv",
            mime="text/csv",
        )

        if len(history_df) > 1:
            st.subheader("Estatísticas Resumidas")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                avg_price = history_df["Predicted Price"].mean()
                st.metric("Preço Médio", f"R$ {avg_price:,.2f}")
            with stat_col2:
                max_price = history_df["Predicted Price"].max()
                st.metric("Maior Preço", f"R$ {max_price:,.2f}")
            with stat_col3:
                min_price = history_df["Predicted Price"].min()
                st.metric("Menor Preço", f"R$ {min_price:,.2f}")

        if st.button("Limpar Histórico"):
            st.session_state.predictions_history = []
            st.rerun()
    else:
        st.info("Ainda não há predições. Use a aba Predição de Preço para fazer sua primeira predição!")

# Rodapé
st.divider()
st.markdown(
    f"""
    <div style='text-align: center; color: #5DCAA5;'>
        <p>Preditor de Preços FIPE | Random Forest</p>
        <p>R² = {metadata['metrics_test']['r2']:.3f} | Treinado com {metadata['train_samples']:,} registros</p>
    </div>
    """,
    unsafe_allow_html=True,
)