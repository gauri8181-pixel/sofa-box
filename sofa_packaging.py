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

# 컨테이너 내부 치수 기본값(mm) - 필요 시 UI에서 수정 가능하게 둠
CONTAINERS = {
    "20ft(표준)": {"L": 5900, "W": 2352, "H": 2395},
    "40ft High Cube": {"L": 12032, "W": 2350, "H": 2700},
}

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

    # 보강재
    # 측면 패드: "넓은면(W×H)" 기준
    side_pad_qty = 2
    side_pad_w = lower_in_w
    side_pad_h = lower_in_h

    # 상부 패드: 상부 외측 W×D
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
# Plotly 3D 도형 유틸
# =========================
def _cuboid_vertices(x0, y0, z0, dx, dy, dz):
    x = [x0, x0+dx, x0+dx, x0,    x0, x0+dx, x0+dx, x0]
    y = [y0, y0,    y0+dy, y0+dy, y0, y0,    y0+dy, y0+dy]
    z = [z0, z0,    z0,    z0,    z0+dz, z0+dz, z0+dz, z0+dz]
    return np.array(x), np.array(y), np.array(z)

def _cuboid_mesh(name, x0, y0, z0, dx, dy, dz, opacity=0.35, color="#888", wire=False, showlegend=True):
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
        showlegend=showlegend,
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

# =========================
# 3D 도면: A1 / 수출형
# =========================
def build_3d_figure_a1(res):
    w = res["A1_외측가로"]
    d = res["A1_외측세로"]
    h = res["A1_외측높이"]

    fig = go.Figure()
    for t in _cuboid_mesh("A1 박스(외측)", 0, 0, 0, w, d, h, opacity=0.35, color="#9C6B3E", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="가로(W)", yaxis_title="세로(D)", zaxis_title="높이(H)", aspectmode="data"),
    )
    return fig

def build_3d_figure_export(res):
    """수출형 3D 분해도 (측면 패드: 넓은면 W×H -> 앞/뒤 면 배치)"""
    lw = res["하부_외측가로"]
    ld = res["하부_외측세로"]
    lh = res["하부_외측높이"]

    uw = res["상부_외측가로"]
    ud = res["상부_외측세로"]
    uh = res["상부_외측높이"]

    gap = max(int(round(lh * 0.15)), 60)

    # 하부
    lower_x0, lower_y0, lower_z0 = 0, 0, 0

    # 측면 패드(넓은면): W×H -> 앞/뒤 배치
    pad_t = 20
    pad_w = res["하부_내측가로"]
    pad_h = res["하부_내측높이"]
    pad_z0 = lower_z0 + gap
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

    for t in _cuboid_mesh("측면 패드(앞)", pad_x0, pad_front_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)
    for t in _cuboid_mesh("측면 패드(뒤)", pad_x0, pad_back_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)

    for idx, (ax0, ay0) in enumerate(angles, start=1):
        for t in _cuboid_mesh(f"앵글({idx}) 70x70", ax0, ay0, 0, ANGLE_W, ANGLE_D, ang_h, opacity=0.55, color="#F2C94C", wire=True):
            fig.add_trace(t)

    for t in _cuboid_mesh("상부 박스(외측)", 0, 0, upper_z0, uw, ud, uh, opacity=0.35, color="#B08968", wire=True):
        fig.add_trace(t)

    for t in _cuboid_mesh("상부 패드(외측 W×D)", 0, 0, top_pad_z0, uw, ud, top_pad_th, opacity=0.6, color="#EB5757", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="가로(W)", yaxis_title="세로(D)", zaxis_title="높이(H)", aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig

# =========================
# 컨테이너 적재 시뮬레이션 (평면 회전만 허용, 세우기/뒤집기 금지)
# =========================
def get_export_package_footprint(res):
    """
    수출형 완성 외형의 바닥 치수(가로/세로)는 상/하부 외측 중 큰 값으로 잡음.
    높이는 C안: 사용자 입력(기본값 자동)으로 처리.
    """
    pack_w = max(res["하부_외측가로"], res["상부_외측가로"])
    pack_d = max(res["하부_외측세로"], res["상부_외측세로"])
    default_h = max(res["하부_외측높이"], res["상부_외측높이"])
    return int(pack_w), int(pack_d), int(default_h)

def compute_packing(container_L, container_W, container_H,
                    pack_w, pack_d, pack_h,
                    gap_xy=10, gap_z=0, margin_L=0, margin_W=0, margin_H=0,
                    allow_rotate=True):
    """
    - 회전은 평면에서만 (w<->d) 허용
    - 세우기/뒤집기(높이 방향 변경) 금지
    """
    def fit(L, W, H, a, b, c):
        # 내부 여유(margin) 반영
        L_eff = max(L - margin_L, 0)
        W_eff = max(W - margin_W, 0)
        H_eff = max(H - margin_H, 0)
        nx = int(np.floor((L_eff + gap_xy) / (a + gap_xy))) if (a + gap_xy) > 0 else 0
        ny = int(np.floor((W_eff + gap_xy) / (b + gap_xy))) if (b + gap_xy) > 0 else 0
        nz = int(np.floor((H_eff + gap_z)  / (c + gap_z)))  if (c + gap_z) > 0 else 0
        return nx, ny, nz, nx*ny*nz

    cand = []
    nx, ny, nz, total = fit(container_L, container_W, container_H, pack_w, pack_d, pack_h)
    cand.append(("기본방향(L=가로, W=세로)", pack_w, pack_d, nx, ny, nz, total))

    if allow_rotate:
        nx2, ny2, nz2, total2 = fit(container_L, container_W, container_H, pack_d, pack_w, pack_h)
        cand.append(("회전방향(L=세로, W=가로)", pack_d, pack_w, nx2, ny2, nz2, total2))

    best = max(cand, key=lambda x: x[-1])
    return best, cand

def build_container_3d(container_L, container_W, container_H,
                       unit_L, unit_W, unit_H,
                       nx, ny, nz,
                       gap_xy=10, gap_z=0,
                       margin_L=0, margin_W=0, margin_H=0,
                       max_display=200):
    """
    컨테이너(와이어) + 박스들(반투명) 3D 표시.
    너무 많으면 max_display까지만 그림.
    """
    fig = go.Figure()

    # 컨테이너 박스(와이어)
    for t in _cuboid_mesh("컨테이너(내부)", 0, 0, 0, container_L, container_W, container_H,
                         opacity=0.05, color="#999", wire=True, showlegend=True):
        fig.add_trace(t)

    # 적재 시작점: margin은 "사용 불가 영역"으로 보고 시작을 margin/2 만큼 안쪽으로 이동(시각화)
    x0 = margin_L / 2
    y0 = margin_W / 2
    z0 = margin_H / 2

    drawn = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if drawn >= max_display:
                    break
                x = x0 + i * (unit_L + gap_xy)
                y = y0 + j * (unit_W + gap_xy)
                z = z0 + k * (unit_H + gap_z)

                # 컨테이너 내부를 넘는 건 그리지 않음(안전)
                if x + unit_L > container_L or y + unit_W > container_W or z + unit_H > container_H:
                    continue

                showleg = (drawn == 0)
                for t in _cuboid_mesh("적재 박스", x, y, z, unit_L, unit_W, unit_H,
                                     opacity=0.25, color="#4A90E2", wire=False, showlegend=showleg):
                    fig.add_trace(t)
                drawn += 1
            if drawn >= max_display:
                break
        if drawn >= max_display:
            break

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="컨테이너 길이(L)", yaxis_title="컨테이너 폭(W)", zaxis_title="컨테이너 높이(H)", aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig, drawn

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
st.title("소파 포장 규격 자동 산정 (A1 도면 + 수출형 3D 분해도 + 엑셀 대량 업로드 + 컨테이너 적재 시뮬레이션)")

tab1, tab2, tab3 = st.tabs(["단건 계산(3D 도해)", "엑셀 대량 업로드", "컨테이너 적재 시뮬레이션(3D)"])

# ---- 탭1 ----
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

# ---- 탭2 ----
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

# ---- 탭3 ----
with tab3:
    st.subheader("컨테이너 적재 시뮬레이션(3D)")
    st.caption("조건: 평면 회전(가로/세로 교환)만 가능 / 세우기·뒤집기 적재 불가(높이 방향 변경 없음)")

    # 탭1에서 계산한 값이 있으면 그걸 쓰고, 없으면 기본값으로 계산
    if "res" in locals():
        res_for_sim = res
    else:
        # 기본값으로 임시 계산
        tmp = calc_all(2000, 900, 800, 30, 30, 30, "DW")
        res_for_sim = {"제품코드": "EX001", "제품명": "소파A", **tmp}

    pack_w, pack_d, pack_h_default = get_export_package_footprint(res_for_sim)

    left, right = st.columns([1, 2], gap="large")
    with left:
        container_name = st.selectbox("컨테이너 선택", list(CONTAINERS.keys()), index=0)
        baseC = CONTAINERS[container_name]
        cL = st.number_input("컨테이너 내부 길이(L, mm)", min_value=1000, value=int(baseC["L"]), step=10)
        cW = st.number_input("컨테이너 내부 폭(W, mm)", min_value=1000, value=int(baseC["W"]), step=10)
        cH = st.number_input("컨테이너 내부 높이(H, mm)", min_value=1000, value=int(baseC["H"]), step=10)

        st.markdown("### 포장품(수출형 완성) 치수")
        st.write(f"- 바닥(자동): **{pack_w} x {pack_d} (mm)**")
        pack_h = st.number_input("완성 높이(H, mm)  [C안: 기본값 자동, 필요 시 수정]", min_value=1, value=int(pack_h_default), step=5)

        allow_rotate = st.checkbox("평면 회전 허용(가로↔세로)", value=True)

        st.markdown("### 적재 여유/간격")
        gap_xy = st.number_input("박스 간 간격(XY, mm)", min_value=0, value=10, step=5)
        gap_z = st.number_input("단 간격(Z, mm)", min_value=0, value=0, step=5)

        margin_L = st.number_input("컨테이너 길이 여유(총, mm)", min_value=0, value=0, step=10)
        margin_W = st.number_input("컨테이너 폭 여유(총, mm)", min_value=0, value=0, step=10)
        margin_H = st.number_input("컨테이너 높이 여유(총, mm)", min_value=0, value=0, step=10)

        max_display = st.slider("3D 표시 최대 박스 수(많으면 느려짐)", min_value=20, max_value=400, value=200, step=10)

        run_sim = st.button("적재 시뮬레이션 실행", type="primary")

    with right:
        if not run_sim:
            st.info("왼쪽에서 컨테이너/높이/간격을 설정하고 **적재 시뮬레이션 실행**을 누르면 3D 배치도와 수량이 나옵니다.")
        else:
            best, cand = compute_packing(
                cL, cW, cH,
                pack_w, pack_d, pack_h,
                gap_xy=gap_xy, gap_z=gap_z,
                margin_L=margin_L, margin_W=margin_W, margin_H=margin_H,
                allow_rotate=allow_rotate
            )

            best_name, unitL, unitW, nx, ny, nz, total = best

            st.subheader("계산 결과")
            st.write(f"**최적 방향:** {best_name}")
            st.write(f"**배치:** {nx}(길이방향) × {ny}(폭방향) × {nz}(높이단) = **총 {total}개**")
            st.write(f"**사용 포장품 치수(적재 기준):** L={unitL}, W={unitW}, H={pack_h} (mm)")

            with st.expander("방향별 비교 보기"):
                df_cand = pd.DataFrame([{
                    "방향": x[0], "적재 L": x[1], "적재 W": x[2],
                    "nx": x[3], "ny": x[4], "nz": x[5], "총수량": x[6]
                } for x in cand])
                st.dataframe(df_cand, use_container_width=True)

            fig, drawn = build_container_3d(
                cL, cW, cH,
                unitL, unitW, pack_h,
                nx, ny, nz,
                gap_xy=gap_xy, gap_z=gap_z,
                margin_L=margin_L, margin_W=margin_W, margin_H=margin_H,
                max_display=max_display
            )
            st.subheader("3D 적재 배치도")
            st.plotly_chart(fig, use_container_width=True)

            if total > drawn:
                st.caption(f"※ 총 {total}개 중 3D는 성능을 위해 {drawn}개까지만 표시했습니다(슬라이더로 조정 가능).")
