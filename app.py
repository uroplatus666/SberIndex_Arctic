import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict

st.set_page_config(layout="wide")

WINTER_CSS = """
<style>
/* Основной фон приложения: северное небо */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b1f33 0%, #020617 55%, #000000 100%);
}

/* Боковая панель — более тёмный градиент */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0b1f33 50%, #020617 100%);
}

/* Шапка — полупрозрачная тёмная полоса */
[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.9);
}

/* Ссылки — холодный голубой оттенок */
.stMarkdown a {
    color: #7CC2DB !important;
}
</style>
"""
st.markdown(WINTER_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_excel('clustered.xlsx')

@st.cache_data
def compute_corr_pairs(df, names, descriptions):
    numeric_cols = list(names.keys())
    corr_matrix = df[numeric_cols].corr()

    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['feature1', 'feature2', 'correlation']

    # оставляем только уникальные пары
    corr_pairs = corr_pairs[corr_pairs['feature1'] < corr_pairs['feature2']]
    corr_pairs = corr_pairs.dropna(subset=['correlation'])

    corr_pairs['abs_correlation'] = np.abs(corr_pairs['correlation'])
    corr_pairs = corr_pairs.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

    corr_pairs['feature1_ru'] = corr_pairs['feature1'].map(names).fillna(corr_pairs['feature1'])
    corr_pairs['feature2_ru'] = corr_pairs['feature2'].map(names).fillna(corr_pairs['feature2'])

    stop_words = ['возраст', 'население', 'зарплата', 'температур', 'образование', 'связность', 'миграц', 'балл']
    pattern = '|'.join(stop_words)

    mask1 = corr_pairs['feature1_ru'].str.contains(pattern, case=False, na=False)
    mask2 = corr_pairs['feature2_ru'].str.contains(pattern, case=False, na=False)

    corr_pairs = corr_pairs[~(mask1 & mask2)]

    corr_pairs['desc1'] = corr_pairs['feature1'].map(descriptions).fillna('Нет описания')
    corr_pairs['desc2'] = corr_pairs['feature2'].map(descriptions).fillna('Нет описания')
    return corr_pairs


@st.cache_data
def build_heating_figure():
    temp_df = pd.read_excel("year_temp.xlsx")

    if "date" in temp_df.columns:
        date_col = "date"
    elif "date_2024" in temp_df.columns:
        date_col = "date_2024"
    else:
        raise KeyError("В year_temp.xlsx нет колонки 'date' или 'date_2024'")

    temp_df = temp_df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = (
        temp_df
        .dropna(subset=[date_col, "T_air", "settlement_name_sep"])
        .sort_values(["settlement_name_sep", date_col])
    )
    temp_df[date_col] = temp_df[date_col].dt.normalize()
    temp_df = temp_df.rename(columns={date_col: "date"})

    all_names = sorted(
        temp_df["settlement_name_sep"].dropna().unique().tolist()
    )
    if not all_names:
        raise ValueError("Нет данных по 'settlement_name_sep' в year_temp.xlsx")

    # --- отопительный период из temp_df ---
    needed_cols = {"spring_off_date", "autumn_on_date", "heating_days_in_year"}
    if not needed_cols.issubset(temp_df.columns):
        raise KeyError(
            "В year_temp.xlsx должны быть колонки "
            "'spring_off_date', 'autumn_on_date', 'heating_days_in_year'"
        )

    _res = temp_df[
        ["settlement_name_sep",
         "spring_off_date",
         "autumn_on_date",
         "heating_days_in_year"]
    ].drop_duplicates().copy()

    _res["spring_off_date"] = pd.to_datetime(
        _res["spring_off_date"], errors="coerce"
    ).dt.normalize()
    _res["autumn_on_date"] = pd.to_datetime(
        _res["autumn_on_date"], errors="coerce"
    ).dt.normalize()

    res_map = _res.set_index("settlement_name_sep")[
        ["spring_off_date", "autumn_on_date", "heating_days_in_year"]
    ].to_dict("index")

    def _trailing_paren_span(s: str):
        j = len(s) - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        if j < 0 or s[j] != ")":
            return None
        depth = 1
        i = j - 1
        while i >= 0:
            c = s[i]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
                if depth == 0:
                    return i, j
            i -= 1
        return None

    def region_of(name: str) -> str:
        span = _trailing_paren_span(name)
        if span:
            i, j = span
            return name[i + 1 : j].strip() or "Не указан"
        return "Не указан"

    def short_label(name: str) -> str:
        span = _trailing_paren_span(name)
        return name[: span[0]].rstrip() if span else name

    # регион -> список поселений ---
    names_by_region = defaultdict(list)
    for nm in all_names:
        names_by_region[region_of(nm)].append(nm)
    for r in list(names_by_region):
        names_by_region[r] = sorted(set(names_by_region[r]))
    regions_sorted = sorted(names_by_region.keys())

    initial_region = regions_sorted[0]
    initial_name = names_by_region[initial_region][0]

    def _year_bounds(name: str):
        g = temp_df[temp_df["settlement_name_sep"] == name]
        if g.empty:
            return None, None
        y = int(g["date"].dt.year.iloc[0])
        return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)

    def _rect(x0, x1):
        return dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor="rgba(215,60,74,0.2)",
            line={"width": 0},
        )

    def heating_shapes(name: str):
        bounds = _year_bounds(name)
        if bounds == (None, None):
            return []
        jan1, dec31 = bounds

        row = res_map.get(name)
        if not row:
            return []

        so = row.get("spring_off_date")
        ao = row.get("autumn_on_date")

        if pd.isna(so) and pd.isna(ao):
            return []

        shapes = []

        # зима с начала года до выключения
        if pd.isna(so) and not pd.isna(ao):
            shapes.append(_rect(jan1, dec31))
            return shapes
        if pd.isna(so):
            return shapes

        last_winter = so - pd.Timedelta(days=1)
        if last_winter >= jan1:
            shapes.append(_rect(jan1, last_winter))

        # осенняя часть
        if not pd.isna(ao) and ao <= dec31:
            shapes.append(_rect(ao, dec31))

        return shapes

    def heating_annotations(name: str):
        row = res_map.get(name)
        if not row:
            return []
        total = row.get("heating_days_in_year")
        if total is None or (isinstance(total, float) and np.isnan(total)):
            return []
        return [
            dict(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.98,
                showarrow=False,
                text=f"Отопительный период: {int(total)} дн.",
                align="left",
                bgcolor="rgba(56,89,148,0.5)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font={"size": 12},
            )
        ]


    def _daily_mean(g_in: pd.DataFrame) -> pd.DataFrame:
        return (
            g_in.groupby("date", as_index=False)["T_air"]
            .mean()
            .sort_values("date")
        )

    try:
        from scipy.interpolate import PchipInterpolator

        def smooth_xy(g_in: pd.DataFrame, n: int = 400):
            g = _daily_mean(g_in)
            if len(g) < 3:
                return g["date"].tolist(), g["T_air"].tolist()

            t = g["date"].view("int64").to_numpy()
            y = g["T_air"].to_numpy()

            p = PchipInterpolator(t, y)
            t_new = np.linspace(t.min(), t.max(), n)
            y_new = p(t_new)

            d = (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(t_new, unit="ns").normalize(),
                        "y": y_new,
                    }
                )
                .groupby("date", as_index=False)
                .mean()
            )
            return d["date"].tolist(), d["y"].tolist()

    except Exception:
        def smooth_xy(g_in: pd.DataFrame, n: int = 400):
            g = _daily_mean(g_in)
            return g["date"].tolist(), g["T_air"].tolist()

    series = {}
    for name, g in temp_df.groupby("settlement_name_sep", sort=True):
        x_s, y_s = smooth_xy(g)
        series[name] = {"x": x_s, "y": y_s}

    def settlement_buttons_for(region: str):
        btns = []
        for name in names_by_region[region]:
            btns.append(
                dict(
                    method="update",
                    label=short_label(name),
                    args=[
                        {
                            "x": [series[name]["x"]],
                            "y": [series[name]["y"]],
                            "name": [name],
                        },
                        {
                            "shapes": heating_shapes(name),
                            "annotations": heating_annotations(name),
                        },
                    ],
                )
            )
        return btns

    def settlement_menu_for(region: str):
        return dict(
            type="dropdown",
            active=0,
            buttons=settlement_buttons_for(region),
            x=0.30,
            y=1.15,
            xanchor="left",
            yanchor="top",
            direction="down",
            pad={"r": 4, "t": 2},
            showactive=False,
        bgcolor="#000000",
        )


    fig = go.Figure(
        go.Scatter(
            x=series[initial_name]["x"],
            y=series[initial_name]["y"],
            mode="lines",
            line_shape="linear",
            name=initial_name,
            line=dict(color="#385994", width=2),
            hovertemplate="<br>День:  %{x|%d.%m}<br>Температура:  %{y:.2f}°C",
        )
    )

    region_buttons = []
    for r in regions_sorted:
        default_settlement = names_by_region[r][0]
        region_buttons.append(
            dict(
                method="update",
                label=r,
                args=[
                    {
                        "x": [series[default_settlement]["x"]],
                        "y": [series[default_settlement]["y"]],
                        "name": [default_settlement],
                    },
                    {
                        "shapes": heating_shapes(default_settlement),
                        "annotations": heating_annotations(default_settlement),
                        "updatemenus[1].buttons": settlement_buttons_for(r),
                        "updatemenus[1].active": 0,
                    },
                ],
            )
        )

    region_menu = dict(
        type="dropdown",
        active=regions_sorted.index(initial_region),
        buttons=region_buttons,
        x=0.00,
        y=1.15,
        xanchor="left",
        yanchor="top",
        direction="down",
        pad={"r": 4, "t": 2},
        showactive=False,
        bgcolor="#000000",
    )

    settlement_menu = settlement_menu_for(initial_region)

    fig.update_layout(
        updatemenus=[region_menu, settlement_menu],
        xaxis_title="День",
        yaxis_title="Температура, °C",
        xaxis=dict(
            type="date",
            tickformat="%d.%m",   # ← на оси X только день и месяц
        ),
        hovermode="x",
        shapes=heating_shapes(initial_name),
        annotations=heating_annotations(initial_name),
        margin=dict(t=90),
	height=550,
    )

    return fig

@st.cache_data
def build_migration_pyramid():
    df = pd.read_excel('migration_for_mo.xlsx')

    # базовая проверка
    need = {"region", "municipality_up_name_actual", "age", "gender", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"В файле нет столбцов: {missing}")

    # чистка и типы
    df = df.dropna(subset=["region", "municipality_up_name_actual", "age", "gender", "value"]).copy()
    df["region"] = df["region"].astype(str)
    df["municipality_up_name_actual"] = df["municipality_up_name_actual"].astype(str)
    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(str)


    # агрегирование на случай дублей строк
    df = (
        df.groupby(["region", "municipality_up_name_actual", "age", "gender"], as_index=False)["value"]
          .sum()
    )

    # список регионов и МО по регионам
    regions_sorted = sorted(df["region"].unique().tolist())
    mos_by_region = {
        r: sorted(df.loc[df["region"] == r, "municipality_up_name_actual"].unique().tolist())
        for r in regions_sorted
    }

    # дефолтный МО для региона: максимальный |суммарный нетто|
    def_muni = (
        df.groupby(["region", "municipality_up_name_actual"])["value"].sum().abs()
          .reset_index()
          .sort_values(["region", "value"], ascending=[True, False])
          .groupby("region").first()["municipality_up_name_actual"].to_dict()
    )

    # предрасчет серий по (регион, МО): возраста и зеркальные значения мужчин/женщин
    series = {}  # key = (region, muni) -> dict(ages, men_x, women_x, customdata_m, customdata_w)
    max_abs = 0.0

    for r in regions_sorted:
        for m in mos_by_region[r]:
            g = df[(df["region"] == r) & (df["municipality_up_name_actual"] == m)].copy()
            if g.empty:
                continue
            # сетка возрастов
            ages = np.arange(g["age"].min(), g["age"].max() + 1)
            piv = (
                g.pivot(index="age", columns="gender", values="value")
                 .reindex(index=ages).fillna(0.0)
            )
            men_vals = piv["Мужчины"].values if "Мужчины" in piv.columns else np.zeros_like(ages, dtype=float)
            wom_vals = piv["Женщины"].values if "Женщины" in piv.columns else np.zeros_like(ages, dtype=float)

            # зеркало: мужчины налево, женщины направо; берём абсолютные величины
            men_x = -np.abs(men_vals)
            wom_x = np.abs(wom_vals)

            series[(r, m)] = {
                "ages": ages,
                "men_x": men_x,
                "women_x": wom_x,
                "cd_m": np.column_stack([[r] * len(ages), [m] * len(ages)]),
                "cd_w": np.column_stack([[r] * len(ages), [m] * len(ages)]),
            }
            max_abs = max(max_abs, float(np.nanmax(np.abs(np.concatenate([men_x, wom_x])))))

    def payload_for(r, m):
        s = series[(r, m)]
        return {
            "x": [s["men_x"], s["women_x"]],
            "y": [s["ages"], s["ages"]],
            "customdata": [s["cd_m"], s["cd_w"]],
            "name": ["Мужчины", "Женщины"],
        }

    M = max_abs if max_abs > 0 else 1
    for cand in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        step = cand
        if M / cand <= 8:
            break
    tick_max = int(np.ceil(M / step) * step)
    tickvals = list(range(-tick_max, tick_max + step, step))
    ticktext = [str(abs(v)) for v in tickvals]

    # начальные значения
    initial_region = regions_sorted[0]
    initial_muni = def_muni.get(initial_region, mos_by_region[initial_region][0])

    s0 = series[(initial_region, initial_muni)]
    bar_width = 0.98

    men_color = "#7CC2DB"
    women_color = "#712025"

    fig = go.Figure(data=[
        go.Bar(
            x=s0["men_x"], y=s0["ages"], orientation="h", name="Мужчины",
            width=[bar_width] * len(s0["ages"]),
            customdata=s0["cd_m"],
            hovertemplate=(
                "Регион: %{customdata[0]}<br>МО: %{customdata[1]}<br>"
                "Возраст: %{y}<br>Мужчины: %{x:.0f}"
            ),
            marker=dict(color=men_color, line=dict(width=0)),
        ),
        go.Bar(
            x=s0["women_x"], y=s0["ages"], orientation="h", name="Женщины",
            width=[bar_width] * len(s0["ages"]),
            customdata=s0["cd_w"],
            hovertemplate=(
                "Регион: %{customdata[0]}<br>МО: %{customdata[1]}<br>"
                "Возраст: %{y}<br>Женщины: %{x:.0f}"
            ),
            marker=dict(color=women_color, line=dict(width=0)),
        ),
    ])

    fig.update_layout(
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
        xaxis=dict(
            range=[-tick_max * 1.1, tick_max * 1.1],
            tickvals=tickvals,
            ticktext=ticktext,
            title="Число людей",
        ),
        yaxis=dict(title="Возраст"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=20, t=88, b=40),
        shapes=[
            dict(
                type="line",
                x0=0, x1=0, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=1),
            )
        ],
    )

    # ----- меню МО для конкретного региона -----
    def muni_buttons_for(region):
        btns = []
        for m in mos_by_region[region]:
            p = payload_for(region, m)
            btns.append(dict(
                method="update",
                label=m,
                args=[
                    {"x": p["x"], "y": p["y"], "customdata": p["customdata"]},
                    {"title": f"Возрастная пирамида: {region} — {m}"},
                ],
            ))
        return btns

    # меню 1: регионы
    region_buttons = []
    for r in regions_sorted:
        m_def = def_muni.get(r, mos_by_region[r][0])
        p = payload_for(r, m_def)
        region_buttons.append(dict(
            method="update",
            label=r,
            args=[
                {"x": p["x"], "y": p["y"], "customdata": p["customdata"]},
                {
                    "title": f"Возрастная пирамида: {r} — {m_def}",
                    "updatemenus[1].buttons": muni_buttons_for(r),
                    "updatemenus[1].active": 0,
                },
            ],
        ))

    region_menu = dict(
        type="dropdown",
        active=regions_sorted.index(initial_region),
        buttons=region_buttons,
        x=0.00, y=1.16, xanchor="left", yanchor="top",
        direction="down", pad={"r": 4, "t": 2},
        showactive=False,
    )

    # меню 2: МО для начального региона
    muni_menu = dict(
        type="dropdown",
        active=mos_by_region[initial_region].index(initial_muni),
        buttons=muni_buttons_for(initial_region),
        x=0.38, y=1.16, xanchor="left", yanchor="top",
        direction="down", pad={"r": 4, "t": 2},
        showactive=False,
    )

    fig.update_layout(updatemenus=[region_menu, muni_menu])

    return fig


@st.cache_data
def load_meta():
    config_path = Path(__file__).parent / "feature_meta.json"
    with config_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return meta["names"], meta["descriptions"]

names, descriptions = load_meta()


df = load_data()
corr_pairs = compute_corr_pairs(df, names, descriptions)

st.title('❄️ Почему одни арктические поселки развиваются быстрее, а другие теряют население?')


top_n = st.slider('Показать топ n пар по модулю корреляции', 10, len(corr_pairs), 50)

st.subheader('Таблица топ-пар корреляций с описаниями')
#st.dataframe(corr_pairs.head(top_n)[['feature1_ru', 'feature2_ru', 'correlation', 'desc1', 'desc2']],
             #column_config={
                 #'feature1_ru': 'Признак 1',
                 #'feature2_ru': 'Признак 2',
                 #'correlation': 'Корреляция',
                 #'desc1': 'Описание 1',
                 #'desc2': 'Описание 2'
             #},
             #width='stretch')

table_df = corr_pairs.head(top_n)[['feature1_ru', 'feature2_ru', 'correlation', 'desc1', 'desc2']].copy()
table_df = table_df.rename(columns={
    'feature1_ru': 'Признак 1',
    'feature2_ru': 'Признак 2',
    'correlation': 'Корреляция',
    'desc1': 'Описание 1',
    'desc2': 'Описание 2'
})

def _corr_color(val):
    if pd.isna(val):
        return ''
    if val < 0:
        return 'background-color: rgba(56, 89, 148, 0.8); color: white;'
    else:
        return 'background-color: rgba(113, 32, 37, 0.8); color: white;'

styled_table = (
    table_df
    .style
    .format({'Корреляция': '{:.2f}'})
    .map(_corr_color, subset=['Корреляция'])
)

col_table, col_space = st.columns([4, 1])
with col_table:
    st.dataframe(styled_table, width='stretch')
st.subheader('🌨️ Миграция и другие показатели')


my_palette = ["#972B34", "#D73C4A", "#7CC2DB", "#5D8CB3"]

numeric_cols = [
 'pop_total',
 'pop_women',
 'pop_men',
 'pop_men_share',
 'avg_age_2023',
 'avg_age_women_2023',
 'avg_age_men_2023',
 'pop_total_rosstat_2023',
 'pop_men_rosstat_2023',
 'pop_women_rosstat_2023',
 'avg_age_2024',
 'avg_age_women_2024',
 'avg_age_men_2024',
 'pop_total_rosstat_2024',
 'pop_men_rosstat_2024',
 'pop_women_rosstat_2024',
 'natural_growth',
 'death_rate',
 'dtp_injury_norm',
 'dtp_deaths_norm',
 # '2023plus_avg_age_women',
 # '2023minus_avg_age_women',
 # '2023migration_women',
 # '2023plus_avg_age_men',
 # '2023minus_avg_age_men',
 # '2023migration_men',
 'POI_num',
 'sport_facilities',
 'sport_halls',
 'sport_pools',
 'sport_stadiums',
 'sport_gym',
 'cinemas',
 'libraries',
 'catering',
 'culture_centers',
 'theatres',
 'retail_federal',
 'grocery store_norm',
 'pvz_norm',
 'highway_dist',
 'railway_dist',
 'ecology_polygon',
 'comments_eco_perc',
 'emissions_second',
 'emissions_first',
 'emissions_all',
 'ecology_projects',
 'ecology_spending',
 'dist_to_airport_geo_km',
 'scheduled_service',
 'type_airport_small',
 'aviation_sun',
 'aviation_center',
 'primary_avaiability',
 'comments_health_perc',
 'support_staff',
 'pediatricians',
 'terapists',
 'dentists',
 'space_per_capita',
 'families_improve',
 'emergency_housing',
 'housing_depreciation',
 'sale_house_count',
 'sale_house_count_pop',
 '1000rub_per_m2_avg',
 'building_construction_space',
 'building_num',
 'certificate_fail',
 'additional_education',
 'spendings_per_schoolar',
 'math_ege_plus',
 'math_ege_base',
 'russian_oge',
 'math_oge',
 'russian_ege',
 'nomadic_education_primary',
 'nomadic_education_main',
 'nomadic_education_high',
 'nomadic_ngo',
 'nomadic_language',
 'Все категории',
 'Здоровье',
 'Маркетплейсы',
 'Общественное питание',
 'Продовольствие',
 'Транспорт',
 'market_access',
 'investments_per_capita',
 'gain_results',
 'loss_results',
 'unemployed',
 'revenue_per_capita',
 'profit_bt_per_capita',
 'vacancies',
 'hhru_vacancies_count',
 'hhru_vacancies_count_pop',
 'Все отрасли',
 'wage_average',
 'min_salary',
 'max_salary',
 'rub_for_life',
 'social_share',
 'kindergarden_salary',
 'school_salary',
 'Административная деятельность',
 'Водоснабжение',
 'Гос. управление и военн. безопасность',
 'Гостиницы и общепит',
 'Добыча полезных ископаемых',
 'Здравоохранение',
 'ИТ и связь',
 'Научная и проф. деятельность',
 'Обрабатывающие производства',
 'Образование',
 'Операции с недвижимостью',
 'Сельское хозяйство',
 'Спорт и досуг',
 'Строительство',
 'Торговля',
 'Транспортировка и хранение',
 'Услуги ЖКХ',
 'Финансы и страхование',
 'Прочие услуги',
 'heating_days_in_year',
 'amplitude',
 'min_summer_temp',
 'max_winter_temp',
 'max_year_temp',
 'min_year_temp',
 'set_year']

ru_numeric_cols = [names.get(col, col) for col in numeric_cols]
selected_ru = st.selectbox('Выберите признак для оси X', ru_numeric_cols)

# Находим оригинальное имя колонки
selected_col = next((k for k, v in names.items() if v == selected_ru), selected_ru)

# Показываем описание
st.markdown(f"{descriptions.get(selected_col, 'Нет описания')}")

# Строим график для выбранной колонки
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Женщины', 'Мужчины'),
    shared_xaxes=True,
    horizontal_spacing=0.1
)

clusters = [0, 1, 2, 3]

for cluster in clusters:
    df_cluster = df[df['cluster'] == cluster].sort_values(selected_col)
    if not df_cluster.empty:
        fake_col = [selected_ru] * len(df_cluster)
        # Для женщин
        fig.add_trace(
            go.Scatter(
                x=df_cluster[selected_col],
                y=df_cluster['2023migration_women'],
                mode='markers',
                name=f'Кластер {cluster}',
                marker=dict(
                    size=8,
                    color=my_palette[cluster]
                ),
                text=df_cluster[['cluster', 'settlement_name_sep']].assign(col_name=fake_col),
                hovertemplate='<b>%{text[1]}</b><br>'
                              '<b>Кластер</b>: %{text[0]}<br>'
                              '<b>%{text[2]}</b>: %{x:.2f}<br>'
                              '<b>Миграция женщин в 2023 г.</b>: %{y}<extra></extra>',
                legendgroup=f'group{cluster}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Для мужчин
        fig.add_trace(
            go.Scatter(
                x=df_cluster[selected_col],
                y=df_cluster['2023migration_men'],
                mode='markers',
                name=f'Кластер {cluster}',
                marker=dict(
                    size=8,
                    color=my_palette[cluster]
                ),
                text=df_cluster[['cluster', 'settlement_name_sep']].assign(col_name=fake_col),
                hovertemplate='<b>%{text[1]}</b><br>'
                              '<b>Кластер</b>: %{text[0]}<br>'
                              '<b>%{text[2]}</b>: %{x:.2f}<br>'
                              '<b>Миграция мужчин в 2023 г.</b>: %{y}<extra></extra>',
                legendgroup=f'group{cluster}',
                showlegend=False
            ),
            row=1, col=2
        )

fig.update_layout(
    yaxis_title='Женщины',
    yaxis2_title='Мужчины',
    xaxis_title=selected_ru,
    xaxis2_title=selected_ru,
    xaxis=dict(
        type='linear',
        nticks=5
    ),
    xaxis2=dict(
        type='linear',
        nticks=5
    ),
    template='plotly_white',
    legend=dict(
        title='Кластеры',
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='left',
        x=1.02
    )
)

col_fig, col_space = st.columns([4, 1])
with col_fig:
    st.plotly_chart(fig, width='stretch')


st.subheader('🧊 Особенности населенный пунктов')

# Дополнительные признаки
additional_cols = [
'2023plus_avg_age_women',
'2023minus_avg_age_women',
'2023migration_women',
'2023plus_avg_age_men',
'2023minus_avg_age_men',
'2023migration_men',
'cluster'
]

# Объединяем списки
all_map_cols = numeric_cols + additional_cols

# Dropdown с русскими названиями для карты
ru_map_cols = [names.get(col, col) for col in all_map_cols]
selected_map_ru = st.selectbox('Выберите признак для карты', ru_map_cols)

# Находим оригинальное имя колонки для карты
selected_map_col = next((k for k, v in names.items() if v == selected_map_ru), selected_map_ru)

# Показываем описание под dropdown
st.markdown(
    f"{descriptions.get(selected_map_col, 'Нет описания')}"
)

# Строим карту (тёмная тема)
fig_map = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    hover_name='settlement_name_sep',
    color=selected_map_col,
    size='pop_total',
    center={'lat': 70, 'lon': 100},
    custom_data=[selected_map_col, 'cluster'],
    color_continuous_scale='RdBu_r',
    template='plotly_dark'
)

# Фиксируем Россию, чтобы карта не была узкой
fig_map.update_geos(
    projection_type="mercator",
    center=dict(lat=70, lon=100),
    lonaxis=dict(range=[10, 190]),
    lataxis=dict(range=[50, 80])
)

fig_map.update_layout(
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showlakes=True,
        lakecolor='#385994',
        showrivers=True,
        rivercolor='#385994',
        showocean=True,
        oceancolor='#385994',
    ),
    coloraxis_colorbar=dict(
        title=selected_map_ru,
        orientation='h',
        x=0.5,
        xanchor='center',
        y=-0.15,
        yanchor='bottom',
        lenmode='fraction',
        len=0.7,
        bgcolor='rgba(0,0,0,0)',
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font_color='#f0f0f0',
    height=550,
    margin=dict(l=0, r=0, t=40, b=80),
)


fig_map.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br>"  # ← сюда попадает hover_name='settlement_name_sep'
        f"{names.get(selected_map_col, selected_map_col)}: " + "%{customdata[0]}<br>"
        "Широта: %{lat}<br>"
        "Долгота: %{lon}<br>"
        "Кластер: %{customdata[1]}<extra></extra>"
    )
)
col_map, col_map_space = st.columns([4, 1])
with col_map:
    st.plotly_chart(fig_map, width='stretch')
    st.caption(
        "Размер кружков на карте пропорционален численности населения населённого пункта."
    )

st.subheader("⛷️ Возрастная пирамида миграции по Муниципальным округам")
fig_migration = build_migration_pyramid()
col_mig, col_mig_space = st.columns([4, 1])
with col_mig:
    st.plotly_chart(fig_migration, width='stretch')

st.subheader("🔥❄️ Годовая динамика температур за последние 10 лет и длина отопительного периода")
fig_heating = build_heating_figure()

col_heat, col_space = st.columns([4, 1])
with col_heat:
    st.plotly_chart(fig_heating, width='stretch')
