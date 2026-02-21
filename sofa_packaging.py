import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# 규칙(고정값)
# =========================
BOX_ADD = {
    "DW": {"add_wd": 10, "add_h": 20},
    "SW": {"add_wd": 6,  "add_h": 12},
}

ANGLE_W = 70
ANGLE_D = 70
LEAVE_H = 50  # 상부 덮개 끝에서 50mm 남김(높이)

# =========================
# 계산 로직
# =========================
def calc_all(base_w, base_d, base_h, mw, md, mh, box_type):
    if box_type not in BOX_ADD:
        raise ValueError("박스형태는 DW 또는 SW여야 합니다.")

    add_wd = BOX_ADD[box_type]["add_wd"]
    add_h = BOX_ADD[box_type]["add_h"]

    # A1 내측 = 제품 내측 + 여유치
    a1_in_w = int(round(base_w + mw))
    a1_in_d = int(round(base_d + md))
    a1_in_h = int(round(base_h + mh))

    # A1 외측
    a1_out_w = a1_in_w + add_wd
    a1_out_d = a1_in_d + add_wd
    a1_out_h = a1_in_h + add_h

    # 수출형 하부
    lower_in_w, lower_in_d, lower_in_h = a1_in_w, a1_in_d, a1_in_h
    lower_out_w, lower_out_d = a1_out_w, a1_out_d
    lower_out_h = lower_in_h + int(round(add_h / 2))  # DW면 +10

    # 수출형 상부
    upper_in_w = a1_in_w + 10
    upper_in_d = a1_in_d + 10
    upper_in_h = a1_in_h - LEAVE_H
    if upper_in_h <= 0:
        upper_in_h = 1

    upper_out_w = upper_in_w + 10
    upper_out_d = upper_in_d + 10
    upper_out_h = upper_in_h + 10

    # =========================
    # 보강재 (변경 반영)
    # 2) 측면 패드는 "넓은쪽 측면" = 가로*높이 면에 들어감
    #    => 규격: 하부 내측 가로(W) * 하부 내측 높이(H)
    # =========================
    side_pad_qty = 2
    side_pad_w = lower_in_w   # 가로(W)
    side_pad_h = lower_in_h   # 높이(H)

    # 상부 패드: 상부 외측 가로*세로
    top_pad_qty = 1
    top_pad_w = upper_out_w
    top_pad_d = upper_out_d

    # 앵글: 70*70 고정, 높이=하부 내측 높이
    angle_qty = 4
    angle_h = lower_in_h

    return {
        "박스형태": box_type,
        "여유가로": mw, "여유세로": md, "여유높이": mh,

        "A1_내측가로": a1_in_w, "A1_내측세로": a1_in_d, "A1_내측높이": a1_in_h,
        "A1_외측가로": a1_out_w, "A1_외측세로": a1_out_d, "A1_외측높이": a1_out_h,

        "하부_내측가로": lower_in_w, "하부_내측세로": lower_in_d, "하부_내측높이": lower_in_h,
        "하부_외측가로": lower_out_w, "하부_외측세로": lower_out_d, "하부_외측높이": lower_out_h,

        "상부_내측가로": upper_in_w, "상부_내측세로": upper_in_d, "상부_내측높이": upper_in_h,
        "상부_외측가로": upper_out_w, "상부_외측세로": upper_out_d, "상부_외측높이": upper_out_h,

        "측면패드_수량": side_pad_qty,
        "측면패드_규격(WxH)": f"{side_pad_w}x{side_pad_h}",

        "상부패드_수량": top_pad_qty,
        "상부패드_규격(WxD)": f"{top_pad_w}x{top_pad_d}",

        "앵글_수량": angle_qty,
        "앵글_규격(70x70xH)": f"{ANGLE_W}x{ANGLE_D}x{angle_h}",
    }

# =========================
# 3D 그리기(분해도) - Plotly
# =========================
def _cuboid_vertices(x0, y0, z0, dx, dy, dz):
    x = [x0, x0+dx, x0+dx, x0,    x0, x0+dx, x0+dx, x0]
    y = [y0, y0,    y0+dy, y0+dy, y0, y0,    y0+dy, y0+dy]
    z = [z0, z0,    z0,    z0,    z0+dz, z0+dz, z0+dz, z0+dz]
    return np.array(x), np.array(y), np.array(z)

def _cuboid_mesh(name, x0, y0, z0, dx, dy, dz, opacity=0.35, color="#888", wire=False):
    x, y, z = _cuboid_vertices(x0, y0, z0, dx, dy, dz)

    i = [0,0,0, 1,1,2, 4,4,4, 5,5,6]
    j = [1,2,3, 2,5,6, 5,6,7, 6,1,2]
    k = [2,3,0, 5,6,7, 6,7,4, 1,2,3]

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name=name,
        opacity=opacity,
        color=color,
        flatshading=True,
        hovertemplate=f"{name}<extra></extra>"
    )
    traces = [mesh]

    if wire:
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        xs, ys, zs = [], [], []
        for a,b in edges:
            xs += [x[a], x[b], None]
            ys += [y[a], y[b], None]
            zs += [z[a], z[b], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=3, color="rgba(0,0,0,0.35)"),
            showlegend=False,
            hoverinfo="skip"
        ))
    return traces

def build_3d_figure_export(res):
    """수출형 3D 분해도 (측면 패드: 넓은면(W×H) -> 앞/뒤 면 배치)"""
    lw = res["하부_외측가로"]
    ld = res["하부_외측세로"]
    lh = res["하부_외측높이"]

    uw = res["상부_외측가로"]
    ud = res["상부_외측세로"]
    uh = res["상부_외측높이"]

    gap = max(int(round(lh * 0.15)), 60)

    # 하부
    lower_x0, lower_y0, lower_z0 = 0, 0, 0

    # 측면 패드(넓은면): W×H -> 앞/뒤(세로 방향 면) 배치
    # 표현용 두께
    pad_t = 20
    pad_w = res["하부_내측가로"]
    pad_h = res["하부_내측높이"]
    pad_z0 = lower_z0 + gap

    # 앞면 패드: y=10 근처, 뒤면 패드: y=ld-pad_t-10
    pad_x0 = (lw - pad_w) / 2
    pad_front_y0 = 10
    pad_back_y0 = ld - pad_t - 10

    # 앵글(4개)
    ang_h = res["하부_내측높이"]
    angles = [
        (0, 0),
        (lw-ANGLE_W, 0),
        (0, ld-ANGLE_D),
        (lw-ANGLE_W, ld-ANGLE_D),
    ]

    # 상부 박스(위로 띄움)
    upper_z0 = lower_z0 + lh + gap * 2

    # 상부 패드(맨 위)
    top_pad_th = 10
    top_pad_z0 = upper_z0 + uh + gap

    fig = go.Figure()

    for t in _cuboid_mesh("하부 박스(외측)", lower_x0, lower_y0, lower_z0, lw, ld, lh, opacity=0.35, color="#A67C52", wire=True):
        fig.add_trace(t)

    # 측면 패드(앞/뒤)
    for t in _cuboid_mesh("측면 패드(앞)", pad_x0, pad_front_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)
    for t in _cuboid_mesh("측면 패드(뒤)", pad_x0, pad_back_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)

    # 앵글
    for idx, (ax0, ay0) in enumerate(angles, start=1):
        for t in _cuboid_mesh(f"앵글({idx}) 70x70", ax0, ay0, 0, ANGLE_W, ANGLE_D, ang_h, opacity=0.55, color="#F2C94C", wire=True):
            fig.add_trace(t)

    # 상부 박스
    for t in _cuboid_mesh("상부 박스(외측)", 0, 0, upper_z0, uw, ud, uh, opacity=0.35, color="#B08968", wire=True):
        fig.add_trace(t)

    # 상부 패드(상부 외측 W×D)
    for t in _cuboid_mesh("상부 패드(외측 W×D)", 0, 0, top_pad_z0, uw, ud, top_pad_th, opacity=0.6, color="#EB5757", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="가로(W)",
            yaxis_title="세로(D)",
            zaxis_title="높이(H)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig

def build_3d_figure_a1(res):
    """A1 3D 도면(외측 박스 1개)"""
    w = res["A1_외측가로"]
    d = res["A1_외측세로"]
    h = res["A1_외측높이"]

    fig = go.Figure()
    for t in _cuboid_mesh("A1 박스(외측)", 0, 0, 0, w, d, h, opacity=0.35, color="#9C6B3E", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="가로(W)",
            yaxis_title="세로(D)",
            zaxis_title="높이(H)",
            aspectmode="data",
        ),
        showlegend=True
    )
    return fig

# =========================
# 엑셀(대량 업로드/다운로드)
# =========================
REQ_COLS = [
    "제품코드", "제품명",
    "내측가로", "내측세로", "내측높이",
    "박스형태", "여유가로", "여유세로", "여유높이"
]

def build_template_excel_bytes():
    df = pd.DataFrame(columns=REQ_COLS)
    df.loc[0] = ["EX001", "소파A", 2000, 900, 800, "DW", 30, 30, 30]
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="입력")
    out.seek(0)
    return out.getvalue()

def process_bulk_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"업로드 엑셀에 필요한 컬럼이 없습니다: {missing}")

    results = []
    for _, row in df.iterrows():
        r = calc_all(
            row["내측가로"], row["내측세로"], row["내측높이"],
            row["여유가로"], row["여유세로"], row["여유높이"],
            str(row["박스형태"]).strip().upper()
        )
        r["제품코드"] = row["제품코드"]
        r["제품명"] = row["제품명"]
        results.append(r)

    out_df = pd.DataFrame(results)
    front = ["제품코드", "제품명"]
    cols = front + [c for c in out_df.columns if c not in front]
    out_df = out_df[cols]

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="원본")
        out_df.to_excel(writer, index=False, sheet_name="결과")
    out.seek(0)
    return out_df, out.getvalue()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="소파 포장 규격 자동 산정", layout="wide")
st.title("소파 포장 규격 자동 산정 (A1 도면 + 수출형 3D 분해도 + 엑셀 대량 업로드)")

tab1, tab2 = st.tabs(["단건 계산(3D 도해)", "엑셀 대량 업로드"])

with tab1:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("입력(한 건)")
        제품코드 = st.text_input("제품코드", value="EX001")
        제품명 = st.text_input("제품명", value="소파A")

        base_w = st.number_input("내측가로", min_value=1, value=2000, step=10)
        base_d = st.number_input("내측세로", min_value=1, value=900, step=10)
        base_h = st.number_input("내측높이", min_value=1, value=800, step=10)

        box_type = st.radio("박스형태", options=["DW", "SW"], horizontal=True, index=0)
        mw = st.number_input("여유가로", min_value=0, value=30, step=5)
        md = st.number_input("여유세로", min_value=0, value=30, step=5)
        mh = st.number_input("여유높이", min_value=0, value=30, step=5)

        run = st.button("계산 실행", type="primary")

    with right:
        if not run:
            st.info("왼쪽 값을 입력하고 **계산 실행**을 누르면, A1 3D 도면 + 수출형 3D 분해도 + 결과표가 표시됩니다.")
        else:
            res = calc_all(base_w, base_d, base_h, mw, md, mh, box_type)
            res = {"제품코드": 제품코드, "제품명": 제품명, **res}
            show_df = pd.DataFrame([res])

            st.subheader("결과(내측/외측 포함)")
            st.dataframe(show_df, use_container_width=True)

            st.subheader("A1 박스 3D 도면(외측)")
            st.plotly_chart(build_3d_figure_a1(res), use_container_width=True)

            st.subheader("수출형 3D 분해도(측면 패드=넓은면 W×H)")
            st.plotly_chart(build_3d_figure_export(res), use_container_width=True)

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                show_df.to_excel(writer, index=False, sheet_name="결과")
            out.seek(0)
            st.download_button(
                "단건 결과 엑셀 다운로드(.xlsx)",
                data=out.getvalue(),
                file_name=f"{제품코드}_포장규격_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab2:
    st.subheader("엑셀 대량 업로드 → 결과 엑셀 다운로드")

    col_a, col_b = st.columns([1, 2], gap="large")

    with col_a:
        st.markdown("### 1) 입력 엑셀 템플릿")
        st.download_button(
            "입력 템플릿 다운로드(.xlsx)",
            data=build_template_excel_bytes(),
            file_name="소파포장_입력템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("### 2) 파일 업로드")
        up = st.file_uploader("엑셀 파일 업로드(.xlsx)", type=["xlsx"])
        st.caption("필수 컬럼(한글): 제품코드, 제품명, 내측가로/세로/높이, 박스형태(DW/SW), 여유가로/세로/높이")

    with col_b:
        if up is None:
            st.info("왼쪽 템플릿 구조와 동일한 엑셀을 업로드하면 결과를 생성합니다.")
        else:
            try:
                out_df, out_bytes = process_bulk_excel(up)
                st.success(f"처리 완료: {len(out_df)}건")
                st.dataframe(out_df, use_container_width=True, height=420)

                st.download_button(
                    "결과 엑셀 다운로드(.xlsx)",
                    data=out_bytes,
                    file_name="소파포장_대량결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(str(e))