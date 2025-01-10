import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import bcrypt  # Importa bcrypt para verificação de senha


# ====================== CONFIG STREAMLIT ======================
st.set_page_config(
    page_title="Centro de Medicina Física e Reabilitação",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ====================== FUNÇÃO DE AUTENTICAÇÃO ======================
def check_password():
    """
    Verifica as credenciais do usuário.
    Retorna True se autenticado, caso contrário False.
    """
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.sidebar.header("🔒 Login")
        username = st.sidebar.text_input("Usuário")
        password = st.sidebar.text_input("Senha", type="password")
        login_button = st.sidebar.button("Entrar")

        if login_button:
            if username == st.secrets["USER1_USERNAME"]:
                # Verifica a senha usando bcrypt
                hashed_password = st.secrets["USER1_PASSWORD"].encode('utf-8')
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    st.session_state['logged_in'] = True
                    st.sidebar.success(f"Bem-vindo, {st.secrets['USER1_NAME']}!")
                else:
                    st.sidebar.error("Senha incorreta.")
            else:
                st.sidebar.error("Nome de usuário incorreto.")

    return st.session_state['logged_in']


# Chama a função de verificação no início do app
if not check_password():
    st.stop()


# ====================== LOGO E TÍTULOS ======================
st.markdown(
    """<div style='text-align: center;'>
       <img src='https://www.cm-barcelos.pt/wp-content/uploads/2018/12/santa-casa23.jpg'
       width='400'/></div>""",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #003366;'>Centro de Medicina Física e Reabilitação</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6A6A6A;'>Dashboard de Assiduidade</h4>", unsafe_allow_html=True)
st.write("")


# ====================== SIDEBAR ================================
st.sidebar.header("Opções")
uploaded_file = st.sidebar.file_uploader("Carregar Ficheiro XLS", type=["xls", "xlsx"])
tolerancia = st.sidebar.number_input("Minutos de Tolerância", min_value=0, max_value=60, value=2, step=1)


# ====================== Funções Auxiliares ======================

def carregar_dados(uploaded_file):
    try:
        dfs = pd.read_html(uploaded_file)
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        colunas_minimas = ['Data', 'N.º Mec.', 'Turnos Previstos']
        for col in colunas_minimas:
            if col not in df.columns:
                st.warning(f"Falta a coluna obrigatória: {col}. Verifique o ficheiro.")
                return None
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o ficheiro: {e}")
        return None


def to_time(hhmm):
    try:
        return datetime.strptime(hhmm, "%H:%M")
    except:
        return None


def verificar_entrada_saida(linha, tolerancia_min=2):
    entrada_prevista = to_time(str(linha.get('Entrada Prevista', '00:00')))
    saida_prevista   = to_time(str(linha.get('Saída Prevista', '00:00')))
    entrada_real     = to_time(str(linha.get('E1', '00:00')))
    saida_real       = to_time(str(linha.get('Saída Real', '00:00')))
    turno_lower = str(linha.get('Turnos Previstos', '')).lower()

    if "fr" in turno_lower:
        # Se é feriado p/ este colaborador, não marca incumprimento.
        return True

    if any(x in turno_lower for x in ["dc", "do", "fe", "bm", "lp"]):
        return True

    tol = timedelta(minutes=tolerancia_min)

    # Caso 1: Entrada Real está '00:00' e Saída Real é válida
    if (entrada_real is None) or (entrada_real.strftime("%H:%M") == '00:00'):
        if saida_real and saida_real >= (saida_prevista - tol):
            return True
        else:
            return False
    # Caso 2: Entrada Real válida
    else:
        if (entrada_real <= (entrada_prevista + tol)) and (saida_real >= (saida_prevista - tol)):
            return True
        else:
            return False


def obter_saida_real(linha):
    import pandas as pd
    # Coletar todas as possíveis saídas
    saidas = []
    for col in ['E1', 'E2', 'E3', 'E4', 'S1', 'S2', 'S3', 'S4']:
        val = linha.get(col, '00:00')
        if pd.notna(val) and val != '00:00':
            saidas.append(val)

    if not saidas:
        return '00:00'

    saida_times = pd.to_datetime(saidas, format="%H:%M", errors='coerce')
    maior_saida = max(saida_times)
    return maior_saida.strftime("%H:%M")


def calcular_metricas_basicas(df):
    periodo_inicio = df['Data'].min()
    periodo_fim = df['Data'].max()
    num_colaboradores = df['N.º Mec.'].nunique()
    num_registos = len(df)
    return periodo_inicio, periodo_fim, num_colaboradores, num_registos


def classificar_dia(turno):
    if isinstance(turno, str):
        turno_lower = turno.lower()
        if any(x in turno_lower for x in ["dc", "do", "fe", "bm", "lp"]):
            return "Ausência"
        else:
            return "Trabalho"
    return "Ausência"


def extrair_horario_tdf(texto_tdf):
    import re
    if not isinstance(texto_tdf, str):
        return (None, None)
    padrao = r'(\d{1,2}:\d{2})'
    horarios = re.findall(padrao, texto_tdf)
    if len(horarios) >= 2:
        entrada = horarios[0]
        saida   = horarios[-1]
        return (entrada, saida)
    return (None, None)


def deduzir_por_picagens(subset_dias_uteis):
    pic_cols = [c for c in ['E1','E2','E3','E4','E5','E6','E7','E8'] if c in subset_dias_uteis.columns]
    tempos = []
    for _, row in subset_dias_uteis.iterrows():
        for c in pic_cols:
            t = to_time(row[c])
            if t is not None:
                tempos.append(t)
    if len(tempos) == 0:
        return (None, None)
    earliest = min(tempos)
    latest   = max(tempos)
    ent = arredondar_entrada_up(earliest)
    sai = arredondar_saida_down(latest)
    return (ent.strftime("%H:%M"), sai.strftime("%H:%M"))


def arredondar_entrada_up(t):
    m = t.minute
    if m == 0:
        return t
    elif 1 <= m <= 30:
        return t.replace(minute=30, second=0, microsecond=0)
    else:
        return (t + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def arredondar_saida_down(t):
    m = t.minute
    if 1 <= m < 30:
        return t.replace(minute=0, second=0, microsecond=0)
    elif 30 <= m < 60:
        return t.replace(minute=30, second=0, microsecond=0)
    return t


def criar_tabela_horarios_por_colaborador(df):
    base_cols = ['N.º Mec.', 'Nome', 'Turnos Previstos', 'Data',
                 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
    base_cols = [c for c in base_cols if c in df.columns]
    df_temp = df[base_cols].copy()
    df_temp['dow'] = df_temp['Data'].dt.weekday
    df_temp['turno_lower'] = df_temp['Turnos Previstos'].astype(str).str.lower()
    ausencias_possiveis = ["bm","lp","fe","dc","do"]
    agrupado = df_temp.groupby(['N.º Mec.', 'Nome'], as_index=False)

    lista_linhas = []
    for (mec, nome), subset_colab in agrupado:
        sem_entrada = '08:00'
        sem_saida   = '17:00'
        sab_entrada = '08:00'
        sab_saida   = '13:00'
        turnos_final = ''

        # Dias úteis
        subset_dias_uteis = subset_colab[subset_colab['dow'] <= 4].copy().sort_values('Data')
        tdf_rows = subset_dias_uteis[subset_dias_uteis['turno_lower'].str.contains('(tdf|tf)', na=False)]
        if len(tdf_rows) > 0:
            row_tdf = tdf_rows.iloc[0]
            entrada, saida = extrair_horario_tdf(row_tdf['Turnos Previstos'])
            if entrada and saida:
                sem_entrada = entrada
                sem_saida   = saida
            else:
                e, s = deduzir_por_picagens(subset_dias_uteis)
                if e and s:
                    sem_entrada, sem_saida = e, s
            turnos_final = row_tdf['Turnos Previstos']
        else:
            e, s = deduzir_por_picagens(subset_dias_uteis)
            if e and s:
                sem_entrada, sem_saida = e, s
            else:
                if len(subset_dias_uteis) > 0:
                    tudo_ausencia = True
                    for _, row_uti in subset_dias_uteis.iterrows():
                        t = str(row_uti['turno_lower'])
                        if not any(a in t for a in ausencias_possiveis):
                            tudo_ausencia = False
                            break
                    if tudo_ausencia:
                        sem_entrada = ''
                        sem_saida   = ''

        # Sábado
        subset_sab = subset_colab[subset_colab['dow'] == 5].copy()
        if len(subset_sab) > 0:
            sab_tdf = subset_sab[subset_sab['turno_lower'].str.contains('(tdf|tf)', na=False)]
            if len(sab_tdf) > 0:
                row_sab = sab_tdf.iloc[0]
                ent_sab, sai_sab = extrair_horario_tdf(row_sab['Turnos Previstos'])
                if ent_sab and sai_sab:
                    sab_entrada = ent_sab
                    sab_saida   = sai_sab
                else:
                    e2, s2 = deduzir_por_picagens(subset_sab)
                    if e2 and s2:
                        sab_entrada, sab_saida = e2, s2
            else:
                e2, s2 = deduzir_por_picagens(subset_sab)
                if e2 and s2:
                    sab_entrada, sab_saida = e2, s2
            if any(subset_sab['turno_lower'].str.contains('|'.join(ausencias_possiveis))):
                sab_entrada = ''
                sab_saida   = ''

        d = {
            'Nome': nome,
            'N.º Mec.': mec,
            'Turnos Previstos': turnos_final,
            'Horário Semanal (Entrada)': sem_entrada,
            'Horário Semanal (Saída)':   sem_saida,
            'Horário Sábado (Entrada)':  sab_entrada,
            'Horário Sábado (Saída)':    sab_saida
        }
        lista_linhas.append(d)

    df_colabs = pd.DataFrame(lista_linhas)
    df_colabs.sort_values(by=["Nome","N.º Mec."], inplace=True)
    return df_colabs


def aplicar_horarios_confirmados_no_df(df_original, df_horarios):
    df_final = df_original.copy()
    df_final['Entrada Prevista'] = '00:00'
    df_final['Saída Prevista']   = '00:00'
    hdic = df_horarios.set_index('N.º Mec.').to_dict('index')

    for i, row in df_final.iterrows():
        mec = row['N.º Mec.']
        dow = row['Data'].weekday() if pd.notna(row['Data']) else None
        if (mec in hdic) and (dow is not None):
            sem_entrada = hdic[mec]['Horário Semanal (Entrada)']
            sem_saida   = hdic[mec]['Horário Semanal (Saída)']
            sab_entrada = hdic[mec]['Horário Sábado (Entrada)']
            sab_saida   = hdic[mec]['Horário Sábado (Saída)']

            if dow <= 4:
                df_final.at[i,'Entrada Prevista'] = sem_entrada if sem_entrada else '00:00'
                df_final.at[i,'Saída Prevista']   = sem_saida if sem_saida else '00:00'
            elif dow == 5:
                df_final.at[i,'Entrada Prevista'] = sab_entrada if sab_entrada else '00:00'
                df_final.at[i,'Saída Prevista']   = sab_saida if sab_saida else '00:00'
            else:
                df_final.at[i,'Entrada Prevista'] = '00:00'
                df_final.at[i,'Saída Prevista']   = '00:00'
        else:
            df_final.at[i,'Entrada Prevista'] = '00:00'
            df_final.at[i,'Saída Prevista']   = '00:00'
    return df_final


def filtrar_dias_possiveis(df):
    df2 = df.copy()
    df2['dow'] = df2['Data'].dt.weekday.fillna(-1)
    df2 = df2[df2['dow'] != 6]
    df2['Turnos Previstos'] = df2['Turnos Previstos'].astype(str)
    ausencias = ["dc","do","fe","bm","lp"]
    df2['eh_aus'] = df2['Turnos Previstos'].str.lower().apply(
        lambda x: any(a in x for a in ausencias)
    )
    df2 = df2[df2['eh_aus'] == False]
    df2.drop(columns=['dow','eh_aus'], inplace=True, errors='ignore')
    return df2


def filtrar_dias_trabalho(df):
    df2 = df.copy()
    df2['dow'] = df2['Data'].dt.weekday.fillna(-1)
    df2 = df2[df2['dow'] != 6]
    df2['Turnos Previstos'] = df2['Turnos Previstos'].astype(str)
    ausencias = ["dc","do","fe","bm","lp"]
    df2['eh_aus'] = df2['Turnos Previstos'].str.lower().apply(
        lambda x: any(a in x for a in ausencias)
    )
    df2 = df2[df2['eh_aus'] == False]

    def has_pic_or_turno(row):
        t = row['Turnos Previstos'].strip()
        if t != '':
            return True
        for c in ['E1','E2','E3','E4','E5','E6','E7','E8']:
            if row.get(c,'00:00') != '00:00':
                return True
        return False

    df2['incl'] = df2.apply(has_pic_or_turno, axis=1)
    df2 = df2[df2['incl'] == True]
    df2.drop(columns=['dow','eh_aus','incl'], inplace=True, errors='ignore')
    return df2


# ====================== LÓGICA PRINCIPAL ======================

# -- Inicie a flag de dashboard, caso não exista
if 'gerar_dashboard' not in st.session_state:
    st.session_state['gerar_dashboard'] = False


if uploaded_file is not None:
    df = carregar_dados(uploaded_file)
    if df is not None:
        # Classificar Tipo de Dia
        df['Tipo de Dia'] = df['Turnos Previstos'].apply(classificar_dia)
        df = df.sort_values(by='Nome', ascending=True).reset_index(drop=True)

        st.markdown("## 1. Confirma/edita os horários dos turnos")
        if 'horarios_confirmados' not in st.session_state:
            st.session_state['horarios_confirmados'] = False

        if 'df_horarios_por_colab' not in st.session_state:
            st.session_state['df_horarios_por_colab'] = criar_tabela_horarios_por_colaborador(df)

        df_editado = st.data_editor(st.session_state['df_horarios_por_colab'], num_rows="dynamic")

        if st.button("Confirmar Horários"):
            st.session_state['df_horarios_por_colab'] = df_editado
            st.session_state['horarios_confirmados'] = True
            st.success("Horários confirmados com sucesso! Agora clique em 'Gerar Dashboard' para ver métricas.")

        # Se horários confirmados, exibe o botão para gerar o dashboard
        if st.session_state['horarios_confirmados']:
            if st.button("Gerar Dashboard"):
                st.session_state['gerar_dashboard'] = True

        # Se gerar_dashboard é True, desenhamos as abas
        if st.session_state['gerar_dashboard']:

            # 1) Aplica horários
            df_final = aplicar_horarios_confirmados_no_df(df, st.session_state['df_horarios_por_colab'])
            # 2) Define Saída Real
            df_final['Saída Real'] = df_final.apply(obter_saida_real, axis=1)

            st.write("### Tabela com todos os registos:")
            st.dataframe(df_final)

            # 3) Dias Possíveis (sem domingo, sem ausências, sem exigir picagem)
            df_base = filtrar_dias_possiveis(df_final)
            # 4) Dias Trabalhados
            df_trabalho = df_final.copy()

            # 5) Agrupamento: Dias Possíveis
            df_possible = df_base.groupby(['N.º Mec.', 'Nome'])['Data'].nunique().reset_index(name='Dias Possíveis')
            # 6) Agrupamento: Dias Trabalhados
            df_worked  = df_trabalho.groupby(['N.º Mec.', 'Nome'])['Data'].nunique().reset_index(name='Dias Trabalhados')
            df_merged = pd.merge(df_possible, df_worked, on=['N.º Mec.', 'Nome'], how='outer').fillna(0)
            df_merged['Dias Possíveis'] = df_merged['Dias Possíveis'].astype(int)
            df_merged['Dias Trabalhados'] = df_merged['Dias Trabalhados'].astype(int)

            # 7) Verificação de cumprimento
            df_trabalho['Cumpriu Horário'] = df_trabalho.apply(lambda row: verificar_entrada_saida(row, tolerancia), axis=1)

            # Definir 'incumprimentos'
            incumprimentos = df_trabalho[df_trabalho['Cumpriu Horário'] == False][
                ['N.º Mec.', 'Nome','Data','E1','Saída Real','Turnos Previstos','Entrada Prevista','Saída Prevista']
            ]

            # 8) Agrupamento para Ranking
            cumprimento_por_func = df_trabalho.groupby(['N.º Mec.','Nome'])['Cumpriu Horário'].agg(['sum','count']).reset_index()
            cumprimento_por_func['Percentual_Cumprimento'] = 100.0 * (cumprimento_por_func['sum'] / cumprimento_por_func['count'])
            cumprimento_por_func.sort_values('Percentual_Cumprimento', ascending=False, inplace=True)

            # Organizar o Dashboard em Abas
            tab1, tab2, tab3 = st.tabs(["Visão Geral", "Visão Colaboradores", "Detalhe dos Incumprimentos"])

            with tab1:
                st.markdown("### Visão Geral")
                # Calcular métricas básicas
                periodo_inicio, periodo_fim, num_colaboradores, num_registos = calcular_metricas_basicas(df_final)

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Período Início", periodo_inicio.strftime('%d/%m/%Y') if pd.notna(periodo_inicio) else '-')
                c2.metric("Período Fim", periodo_fim.strftime('%d/%m/%Y') if pd.notna(periodo_fim) else '-')
                c3.metric("Nº Colaboradores", num_colaboradores)
                c4.metric("Nº Registos", num_registos)
                c5.metric("Incumprimentos", incumprimentos.shape[0])
                c6.metric("Percentagem de Assiduidade", f"{cumprimento_por_func['Percentual_Cumprimento'].mean():.2f}%")

                def determinar_status(row):
                    turno_val = row.get('Turnos Previstos', '')
                    cumpriu = row.get('Cumpriu Horário', False)
                    data = row.get('Data', None)

                    if isinstance(data, pd.Timestamp):
                        is_saturday = data.weekday() == 5
                        is_sunday = data.weekday() == 6
                    else:
                        is_saturday = False
                        is_sunday = False

                    if pd.notna(turno_val):
                        turno = str(turno_val).lower()
                    else:
                        turno = ''

                    # Nova regra: se for fr, é feriado => sem incumprimento
                    if 'fr' in turno:
                        return 'Feriado'
                    elif (is_saturday or is_sunday):
                        # Se for sábado ou domingo e 'Turnos Previstos' vazio => DC
                        if str(turno).strip() == '':
                            return 'DC'

                    # Regras já existentes
                    if 'fe' in turno:
                        return 'FE'
                    elif 'bm' in turno:
                        return 'BM'
                    elif 'lp' in turno:
                        return 'LP'
                    elif 'dc' in turno and (is_saturday or is_sunday):
                        return 'DC'
                    elif cumpriu:
                        return 'Cumprimento'
                    else:
                        return 'Incumprimento'

                df_trabalho['Status'] = df_trabalho.apply(determinar_status, axis=1)

            with tab2:
                st.markdown("### Detalhamento por Colaborador")
                mask_working = df_trabalho['Status'].isin(['Cumprimento', 'Incumprimento'])
                df_worked = df_trabalho[mask_working]

                cumprimento_por_func = df_worked.groupby(['N.º Mec.', 'Nome'])['Cumpriu Horário'].agg(['sum','count']).reset_index()
                cumprimento_por_func.rename(columns={
                    'sum': 'Dias em Cumprimento',
                    'count': 'Dias Trabalhados'
                }, inplace=True)

                cumprimento_por_func['Percentagem_Cumprimento'] = cumprimento_por_func.apply(
                    lambda row: f"{(100.0 * row['Dias em Cumprimento'] / row['Dias Trabalhados']):.2f}%"
                    if row['Dias Trabalhados'] > 0 else '-', axis=1
                )

                cumprimento_por_func.sort_values('Percentagem_Cumprimento', ascending=False, inplace=True)

                # Mesclar df_possible com cumprimento_por_func
                df_merged = pd.merge(
                    df_possible, 
                    cumprimento_por_func, 
                    on=['N.º Mec.', 'Nome'], 
                    how='left'
                ).fillna({
                    'Dias em Cumprimento': 0,
                    'Dias Trabalhados': 0,
                    'Percentagem_Cumprimento': '-'
                })

                df_merged['Dias em Cumprimento'] = df_merged['Dias em Cumprimento'].astype(int)
                df_merged['Dias Trabalhados'] = df_merged['Dias Trabalhados'].astype(int)

                df_detalhamento = df_merged[['Nome', 'N.º Mec.', 'Dias em Cumprimento', 'Dias Trabalhados', 'Percentagem_Cumprimento']].copy()
                df_detalhamento.rename(columns={
                    'Nome': 'NOME',
                    'N.º Mec.': 'Nº MECANOGRAFICO',
                    'Dias em Cumprimento': 'DIAS EM CUMPRIMENTOS',
                    'Dias Trabalhados': 'TOTAL DE DIAS DE TRABALHO',
                    'Percentagem_Cumprimento': 'PERCENTAGEM DE CUMPRIMENTO'
                }, inplace=True)

                df_detalhamento = df_detalhamento.sort_values(by='NOME', ascending=True).reset_index(drop=True)
                df_detalhamento['PERCENTAGEM DE CUMPRIMENTO'] = df_detalhamento['PERCENTAGEM DE CUMPRIMENTO'].replace({'nan': '-', 'inf': '-', '-': '-'})
                st.dataframe(df_detalhamento)

            with tab3:
                st.markdown("### Detalhes de Incumprimentos")

                incumprimentos = df_trabalho[df_trabalho['Cumpriu Horário'] == False][
                    ['N.º Mec.', 'Nome', 'Data', 'E1', 'Saída Real', 'Entrada Prevista', 'Saída Prevista']
                ].copy()

                incumprimentos.rename(columns={
                    'Nome': 'NOME',
                    'N.º Mec.': 'Nº MECANOGRAFICO',
                    'Data': 'DATA',
                    'E1': 'ENTRADA REAL',
                    'Saída Real': 'SAÍDA REAL',
                    'Entrada Prevista': 'ENTRADA PREVISTA',
                    'Saída Prevista': 'SAÍDA PREVISTA'
                }, inplace=True)

                incumprimentos['DATA'] = pd.to_datetime(incumprimentos['DATA']).dt.strftime('%Y-%m-%d')
                incumprimentos = incumprimentos.sort_values(by='NOME').reset_index(drop=True)

                if 'df_incs' not in st.session_state:
                    incumprimentos['DESCARTAR'] = False
                    st.session_state['df_incs'] = incumprimentos
                else:
                    novos_incs = incumprimentos[['Nº MECANOGRAFICO','NOME','DATA','ENTRADA REAL','SAÍDA REAL','ENTRADA PREVISTA','SAÍDA PREVISTA']]
                    if 'DESCARTAR' not in st.session_state['df_incs'].columns:
                        st.session_state['df_incs']['DESCARTAR'] = False

                    novos_incs['DESCARTAR'] = False
                    st.session_state['df_incs'] = novos_incs

                df_incs_editor = st.data_editor(
                    st.session_state['df_incs'],
                    key='df_incs_editor',
                    use_container_width=True
                )

                st.session_state['df_incs'] = df_incs_editor

                st.write("#### Linhas marcadas para descartar:")
                descartados = df_incs_editor[df_incs_editor['DESCARTAR'] == True]
                st.dataframe(descartados)

                if st.button("Submeter"):
                    df_incs_remanescentes = df_incs_editor[df_incs_editor['DESCARTAR'] == False].copy()
                    st.session_state['df_incs'] = df_incs_remanescentes
                    st.success("Linhas marcadas como descartadas foram removidas da lista de incumprimentos!")

                    
                    if 'DESCARTAR' in df_incs_remanescentes.columns:
                        df_incs_remanescentes = df_incs_remanescentes.drop(columns=['DESCARTAR'])

                    st.write("### Incumprimentos após Análise:")
                    st.dataframe(df_incs_remanescentes)

    else:
        st.error("DF vazio ou não processado.")
else:
    st.info("Carregue um ficheiro na barra lateral, por favor.")
