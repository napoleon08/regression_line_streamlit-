import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# 페이지 기본 설정
# =========================================================

st.set_page_config(
    page_title="통계적 회귀 분석과 기계학습 회귀 분석",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
    color: #111111;
}

.main {
    background-color: #ffffff;
}

h1, h2, h3, h4, h5, p, label, div, span {
    color: #111111;
}

.title-box {
    background: #f5f7fa;
    border: 1px solid #d9dee5;
    border-radius: 14px;
    padding: 25px;
    margin-bottom: 25px;
}

.card {
    background: #ffffff;
    border: 1px solid #d9dee5;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}

.formula {
    background: #f7f7f7;
    border-left: 5px solid #333333;
    padding: 15px;
    border-radius: 8px;
    font-size: 18px;
    margin: 12px 0;
}

.result {
    background: #f5f7fa;
    border: 1px solid #d9dee5;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.small-text {
    font-size: 14px;
    color: #555555 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 제목
# =========================================================

st.markdown("""
<div class="title-box">

<h1>📈 통계적 회귀 분석과 기계학습 회귀 분석</h1>

<p>
데이터의 관계를 분석하고 회귀선을 구한 뒤,
기계학습의 회귀 방법과 비교해 보는 데이터 분석 프로젝트
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 데이터
# =========================================================

x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([41, 30, 28, 22, 12, 11], dtype=float)

x_mean = np.mean(x)
y_mean = np.mean(y)

# 최소제곱법
a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - a * x_mean

y_pred = a * x + b

error = y - y_pred
error_square = error ** 2

sse = np.sum(error_square)

# =========================================================
# 메뉴
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 오차 계산",
    "② 회귀선 직접 계산",
    "③ 기계학습 회귀",
    "④ 방법 비교",
    "⑤ 확인 퀴즈"
])


# =========================================================
# ① 오차 계산
# =========================================================

with tab1:

    st.header("① 오차와 오차² 계산")

    st.markdown("""
    <div class="card">

    실제 데이터와 회귀선으로 예측한 값의 차이를
    <b>오차(error)</b>라고 한다.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula">
    오차 = 실제값 - 예측값
    <br><br>
    오차² = (실제값 - 예측값)²
    </div>
    """, unsafe_allow_html=True)

    df_error = pd.DataFrame({
        "x": x.astype(int),
        "실제 y": y,
        "예측값 ŷ": np.round(y_pred, 2),
        "오차": np.round(error, 2),
        "오차²": np.round(error_square, 2)
    })

    st.dataframe(
        df_error,
        use_container_width=True,
        hide_index=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("평균 x", f"{x_mean:.1f}")

    with col2:
        st.metric("평균 y", f"{y_mean:.1f}")

    with col3:
        st.metric("오차²의 합", f"{sse:.2f}")

    st.markdown("""
    <div class="card">

    회귀선은 각 데이터의 오차를 단순히 더하는 것이 아니라
    <b>오차를 제곱한 값의 합</b>이 가장 작아지도록 결정한다.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ② 회귀선 직접 계산
# =========================================================

with tab2:

    st.header("② 최소제곱법으로 회귀선 구하기")

    st.write("주어진 자전거 데이터를 이용하여 회귀선을 직접 계산한다.")

    df_calc = pd.DataFrame({
        "x": x.astype(int),
        "y": y.astype(int),
        "x - x̄": np.round(x - x_mean, 2),
        "y - ȳ": np.round(y - y_mean, 2),
        "(x-x̄)(y-ȳ)": np.round((x-x_mean)*(y-y_mean), 2),
        "(x-x̄)²": np.round((x-x_mean)**2, 2)
    })

    st.dataframe(
        df_calc,
        use_container_width=True,
        hide_index=True
    )

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    st.markdown("""
    <div class="formula">

    <b>기울기 a</b>

    <br><br>

    a =
    Σ(x - x̄)(y - ȳ)
    /
    Σ(x - x̄)²

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="result">Σ(x-x̄)(y-ȳ)<br>{numerator:.1f}</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="result">Σ(x-x̄)²<br>{denominator:.1f}</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="result">기울기 a<br>{a:.1f}</div>',
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class="formula">

    <b>절편 b</b>

    <br><br>

    b = ȳ - ax̄

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="result">b = {y_mean:.1f} - ({a:.1f} × {x_mean:.1f}) = {b:.1f}</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 최종 회귀선")

    st.markdown(
        f"""
        <div class="result">

        ŷ = {a:.0f}x + {b:.0f}

        </div>
        """,
        unsafe_allow_html=True
    )

    # 그래프
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="실제 데이터",
            marker=dict(size=11)
        )
    )

    line_x = np.linspace(0.5, 6.5, 100)
    line_y = a * line_x + b

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="회귀선",
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x_mean],
            y=[y_mean],
            mode="markers",
            name="평균점",
            marker=dict(size=14, symbol="diamond")
        )
    )

    fig.update_layout(
        title="자전거 데이터와 회귀선",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"평균점은 ({x_mean:.1f}, {y_mean:.1f})이고, "
        f"회귀선은 ŷ = {a:.0f}x + {b:.0f}이다."
    )


# =========================================================
# ③ 기계학습 회귀
# =========================================================

with tab3:

    st.header("③ 기계학습을 이용한 회귀")

    st.markdown("""
    <div class="card">

    통계적 회귀에서는 수학식을 이용하여
    데이터 전체를 가장 잘 설명하는 회귀선을 구한다.

    기계학습 회귀에서는 학습 데이터를 이용하여
    새로운 데이터의 값을 예측하는 모델을 만든다.

    </div>
    """, unsafe_allow_html=True)

    # 놀이공원 데이터
    hours = np.arange(10, 21)

    waiting = np.array([
        5, 10, 15,
        40, 45, 50, 45, 35, 25,
        15, 5
    ])

    df_amusement = pd.DataFrame({
        "시간": hours,
        "대기시간": waiting
    })

    st.subheader("🎢 놀이공원 데이터")

    st.dataframe(
        df_amusement,
        use_container_width=True,
        hide_index=True
    )

    fig2 = px.scatter(
        df_amusement,
        x="시간",
        y="대기시간",
        title="놀이공원 시간에 따른 대기시간"
    )

    fig2.update_traces(marker=dict(size=12))

    fig2.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # Decision Tree
    # -----------------------------------------------------

    st.subheader("🌳 의사결정나무 회귀")

    st.write(
        "시간을 몇 개의 구간으로 나누고 각 구간의 평균 대기시간을 이용하여 예측하는 방식이다."
    )

    tree_data = pd.DataFrame({
        "구간": [
            "10~12시",
            "13~17시",
            "18~20시"
        ],
        "평균 대기시간": [
            np.mean(waiting[0:3]),
            np.mean(waiting[3:8]),
            np.mean(waiting[8:11])
        ]
    })

    st.dataframe(
        tree_data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # KNN
    # -----------------------------------------------------

    st.subheader("👥 K-최근접 이웃 회귀")

    target_time = st.slider(
        "예측할 시간을 선택하세요.",
        min_value=10,
        max_value=20,
        value=14
    )

    k = st.slider(
        "이웃의 개수 k",
        min_value=1,
        max_value=9,
        value=3
    )

    distances = np.abs(hours - target_time)

    nearest_indices = np.argsort(distances)[:k]

    prediction = np.mean(waiting[nearest_indices])

    nearest_df = pd.DataFrame({
        "시간": hours[nearest_indices],
        "대기시간": waiting[nearest_indices],
        "거리": distances[nearest_indices]
    })

    st.write("가장 가까운 데이터:")

    st.dataframe(
        nearest_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        f"""
        <div class="result">

        {target_time}시의 예상 대기시간

        <br><br>

        약 {prediction:.1f}분

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <b>KNN 회귀의 핵심</b><br><br>

    새로운 데이터와 가까운 데이터를 찾아
    그 값들을 이용해 예측한다.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ④ 방법 비교
# =========================================================

with tab4:

    st.header("④ 통계적 회귀와 기계학습 회귀 비교")

    comparison = pd.DataFrame({
        "구분": [
            "목적",
            "기본 원리",
            "대표적인 방법",
            "결과",
            "새로운 데이터 예측"
        ],

        "통계적 회귀": [
            "변수 사이의 관계를 설명",
            "수학적 회귀식 계산",
            "최소제곱법",
            "회귀식",
            "가능"
        ],

        "기계학습 회귀": [
            "새로운 데이터 예측",
            "데이터에서 패턴 학습",
            "의사결정나무, KNN 등",
            "학습된 모델",
            "가능"
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="card">

    ### 핵심 정리

    <b>통계적 회귀</b>

    → 데이터의 관계를 수학적으로 설명하는 데 초점을 둔다.

    <br><br>

    <b>기계학습 회귀</b>

    → 데이터를 학습하여 새로운 데이터의 값을 예측하는 데 초점을 둔다.

    <br><br>

    두 방법 모두 데이터를 이용하여
    변수 사이의 관계를 분석하고 값을 예측할 수 있다.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ⑤ 퀴즈
# =========================================================

with tab5:

    st.header("⑤ 확인 퀴즈")

    st.write("배운 내용을 확인해 보세요.")

    questions = [

        {
            "question": "1. 회귀선에서 실제값과 예측값의 차이를 무엇이라고 하는가?",
            "options": [
                "평균",
                "오차",
                "기울기",
                "절편"
            ],
            "answer": "오차"
        },

        {
            "question": "2. 최소제곱법에서는 무엇을 최소화하는가?",
            "options": [
                "x의 평균",
                "y의 평균",
                "오차 제곱의 합",
                "데이터의 개수"
            ],
            "answer": "오차 제곱의 합"
        },

        {
            "question": "3. 주어진 데이터의 회귀선은 무엇인가?",
            "options": [
                "ŷ = 6x + 45",
                "ŷ = -6x + 45",
                "ŷ = -45x + 6",
                "ŷ = 45x - 6"
            ],
            "answer": "ŷ = -6x + 45"
        },

        {
            "question": "4. KNN 회귀에서 예측에 사용하는 것은?",
            "options": [
                "가장 먼 데이터",
                "무작위 데이터",
                "가까운 데이터",
                "평균점 하나"
            ],
            "answer": "가까운 데이터"
        },

        {
            "question": "5. 의사결정나무 회귀에서 데이터를 나누는 기준은?",
            "options": [
                "조건과 분기",
                "무작위 선택",
                "문자 수",
                "파일 크기"
            ],
            "answer": "조건과 분기"
        }
    ]

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    answers = {}

    for i, q in enumerate(questions):

        st.markdown("---")

        st.subheader(q["question"])

        answers[i] = st.radio(
            "답을 선택하세요.",
            q["options"],
            key=f"question_{i}"
        )

    if st.button("정답 확인", type="primary"):

        score = 0

        for i, q in enumerate(questions):

            if answers[i] == q["answer"]:
                score += 1

        st.session_state.quiz_submitted = True
        st.session_state.quiz_score = score

    if st.session_state.quiz_submitted:

        score = st.session_state.quiz_score

        st.markdown("---")

        st.subheader("📊 결과")

        st.markdown(
            f"""
            <div class="result">

            {score} / {len(questions)} 문제 정답

            </div>
            """,
            unsafe_allow_html=True
        )

        for i, q in enumerate(questions):

            if answers[i] == q["answer"]:
                st.success(
                    f"{i+1}번: 정답입니다. → {q['answer']}"
                )
            else:
                st.error(
                    f"{i+1}번: 오답입니다. 정답 → {q['answer']}"
                )


# =========================================================
# 하단
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;">

<p class="small-text">

통계적 회귀 분석 · 최소제곱법 · 기계학습 회귀 분석

</p>

</div>
""", unsafe_allow_html=True)
