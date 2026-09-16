import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import html


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
POSTER_DIR = os.path.join(BASE_DIR, "posters")

os.makedirs(POSTER_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    with open(
        os.path.join(MODEL_DIR, "vectorizer.pkl"),
        "rb"
    ) as file:
        vectorizer = pickle.load(file)

    with open(
        os.path.join(MODEL_DIR, "feature_matrix.pkl"),
        "rb"
    ) as file:
        feature_matrix = pickle.load(file)

    with open(
        os.path.join(MODEL_DIR, "similarity_matrix.pkl"),
        "rb"
    ) as file:
        similarity_matrix = pickle.load(file)

    with open(
        os.path.join(MODEL_DIR, "recommendation_df.pkl"),
        "rb"
    ) as file:
        recommendation_df = pickle.load(file)

    with open(
        os.path.join(MODEL_DIR, "movie_indices.pkl"),
        "rb"
    ) as file:
        movie_indices = pickle.load(file)

    with open(
        os.path.join(MODEL_DIR, "mood_mapping.pkl"),
        "rb"
    ) as file:
        mood_mapping = pickle.load(file)

    return (
        vectorizer,
        feature_matrix,
        similarity_matrix,
        recommendation_df,
        movie_indices,
        mood_mapping
    )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    (
        vectorizer,
        feature_matrix,
        similarity_matrix,
        recommendation_df,
        movie_indices,
        mood_mapping
    ) = load_artifacts()

except Exception:

    st.error(
        "Unable to load the movie recommendation system."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False


# ============================================================
# POSTER FUNCTIONS
# ============================================================

def get_poster_path(movie_id):

    try:
        movie_id = int(movie_id)

    except Exception:
        return None

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]

    for extension in extensions:

        poster_path = os.path.join(
            POSTER_DIR,
            f"{movie_id}{extension}"
        )

        if os.path.exists(poster_path):
            return poster_path

    return None


def poster_exists(movie_id):

    return get_poster_path(movie_id) is not None


# ============================================================
# MOVIE HELPERS
# ============================================================

def get_year(movie):

    if "release_year" not in movie.index:
        return "N/A"

    year = movie["release_year"]

    if pd.isna(year):
        return "N/A"

    try:
        return str(int(year))

    except Exception:
        return "N/A"


def get_language(movie):

    if "original_language" not in movie.index:
        return "N/A"

    language = movie["original_language"]

    if pd.isna(language):
        return "N/A"

    return str(language).upper()


def get_genres(movie):

    if "genres_list" not in movie.index:
        return []

    genres = movie["genres_list"]

    if not isinstance(genres, list):
        return []

    return [
        str(genre)
        .replace("_", " ")
        .title()
        for genre in genres
        if genre
    ]


def safe_text(value):

    if pd.isna(value):
        return ""

    return html.escape(str(value))


# ============================================================
# COMMON ITEMS
# ============================================================

def get_common_items(list1, list2):

    if not isinstance(list1, list):
        list1 = []

    if not isinstance(list2, list):
        list2 = []

    set1 = {
        str(item).lower().strip()
        for item in list1
        if item
    }

    set2 = {
        str(item).lower().strip()
        for item in list2
        if item
    }

    return list(set1.intersection(set2))


# ============================================================
# EXPLAIN RECOMMENDATION
# ============================================================

def explain_recommendation(
    source_movie,
    recommended_movie
):

    reasons = []

    # Similar Genres
    if (
        "genres_list" in source_movie.index
        and
        "genres_list" in recommended_movie.index
    ):

        common_genres = get_common_items(
            source_movie["genres_list"],
            recommended_movie["genres_list"]
        )

        if common_genres:
            reasons.append(
                "🎭 Similar genres"
            )

    # Similar Themes
    if (
        "keywords_list" in source_movie.index
        and
        "keywords_list" in recommended_movie.index
    ):

        common_keywords = get_common_items(
            source_movie["keywords_list"],
            recommended_movie["keywords_list"]
        )

        if common_keywords:
            reasons.append(
                "🏷️ Similar themes"
            )

    # Shared Cast
    if (
        "cast_names" in source_movie.index
        and
        "cast_names" in recommended_movie.index
    ):

        common_cast = get_common_items(
            source_movie["cast_names"],
            recommended_movie["cast_names"]
        )

        if common_cast:
            reasons.append(
                "👥 Shared cast"
            )

    # Same Director
    if (
        "director" in source_movie.index
        and
        "director" in recommended_movie.index
    ):

        director_1 = str(
            source_movie["director"]
        ).lower().strip()

        director_2 = str(
            recommended_movie["director"]
        ).lower().strip()

        if (
            director_1
            and
            director_1 != "nan"
            and
            director_1 == director_2
        ):

            reasons.append(
                "🎬 Same director"
            )

    # Content Similarity
    similarity = float(
        recommended_movie["similarity_score"]
    )

    if similarity >= 0.70:

        reasons.append(
            "🧠 Strong content similarity"
        )

    elif similarity >= 0.40:

        reasons.append(
            "🧠 Similar movie content"
        )

    if not reasons:

        reasons.append(
            "✨ Similar overall movie profile"
        )

    return reasons


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_movies(
    movie_title,
    top_n=10
):

    selected_rows = recommendation_df[
        recommendation_df["title"] == movie_title
    ]

    if selected_rows.empty:
        return pd.DataFrame()

    movie_index = selected_rows.index[0]

    similarity_scores = np.asarray(
        similarity_matrix[movie_index]
    ).flatten()

    candidates = recommendation_df.copy()

    candidates["similarity_score"] = similarity_scores

    candidates = candidates[
        candidates["title"] != movie_title
    ].copy()

    # Rating
    if "rating_score" in candidates.columns:

        rating_score = candidates[
            "rating_score"
        ]

    else:

        rating_score = (
            candidates["vote_average"] / 10
        )

    # Popularity
    if "popularity_score" in candidates.columns:

        popularity_score = candidates[
            "popularity_score"
        ]

    else:

        pop_min = recommendation_df[
            "popularity"
        ].min()

        pop_max = recommendation_df[
            "popularity"
        ].max()

        if pop_max > pop_min:

            popularity_score = (
                candidates["popularity"] - pop_min
            ) / (
                pop_max - pop_min
            )

        else:

            popularity_score = 0

    # Vote confidence
    if "vote_confidence" in candidates.columns:

        vote_confidence = candidates[
            "vote_confidence"
        ]

    else:

        vote_confidence = np.log1p(
            candidates["vote_count"]
        )

        vote_min = vote_confidence.min()
        vote_max = vote_confidence.max()

        if vote_max > vote_min:

            vote_confidence = (
                vote_confidence - vote_min
            ) / (
                vote_max - vote_min
            )

        else:

            vote_confidence = 0

    # Hybrid score
    candidates["hybrid_score"] = (
        0.70 * candidates["similarity_score"]
        +
        0.15 * rating_score
        +
        0.10 * popularity_score
        +
        0.05 * vote_confidence
    )

    # Remove duplicate titles
    candidates = candidates.drop_duplicates(
        subset="title",
        keep="first"
    )

    # Only poster available
    candidates["poster_available"] = (
        candidates["id"].apply(poster_exists)
    )

    candidates = candidates[
        candidates["poster_available"]
    ].copy()

    candidates = candidates.sort_values(
        "hybrid_score",
        ascending=False
    )

    return candidates.head(top_n)


# ============================================================
# ============================================================
# BRIGHT PREMIUM CINEMATIC CSS
# ============================================================

st.markdown(
"""
<style>

/* =========================================================
   GLOBAL BACKGROUND
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(99,102,241,0.26),
            transparent 30%
        ),
        radial-gradient(
            circle at 100% 0%,
            rgba(168,85,247,0.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(34,211,238,0.13),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #121936 0%,
            #11172d 45%,
            #17152f 100%
        );

    color: #f8fafc;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   HERO
   ========================================================= */

.hero-box {
    position: relative;
    overflow: hidden;
    text-align: center;

    padding: 60px 30px;
    margin-bottom: 40px;

    border-radius: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(67,56,145,0.98),
            rgba(45,58,130,0.97),
            rgba(103,55,128,0.96)
        );

    border:
        1px solid rgba(196,181,253,0.34);

    box-shadow:
        0 24px 70px rgba(15,23,42,0.42),
        inset 0 1px 0 rgba(255,255,255,0.08);
}

.hero-box::before {
    content: "";
    position: absolute;

    width: 360px;
    height: 360px;

    top: -190px;
    left: -80px;

    border-radius: 50%;

    background: rgba(96,165,250,0.30);

    filter: blur(60px);
}

.hero-box::after {
    content: "";
    position: absolute;

    width: 360px;
    height: 360px;

    bottom: -200px;
    right: -90px;

    border-radius: 50%;

    background: rgba(244,114,182,0.24);

    filter: blur(65px);
}

.hero-icon {
    position: relative;
    z-index: 2;

    font-size: 52px;
    margin-bottom: 10px;
}

.hero-title {
    position: relative;
    z-index: 2;

    font-size: 52px;
    font-weight: 900;

    letter-spacing: -1.8px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #e0e7ff,
            #a5f3fc,
            #fbcfe8
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    position: relative;
    z-index: 2;

    margin-top: 14px;

    color: #eef2ff;

    font-size: 18px;
    font-weight: 500;
}


/* =========================================================
   SEARCH
   ========================================================= */

.search-heading {
    font-size: 24px;
    font-weight: 850;

    color: #ffffff;

    margin-top: 15px;
    margin-bottom: 11px;
}

div[data-baseweb="select"] > div {
    background:
        linear-gradient(
            135deg,
            #253052,
            #252b55
        ) !important;

    border:
        1px solid rgba(165,180,252,0.38) !important;

    border-radius: 14px !important;

    min-height: 52px;

    box-shadow:
        0 8px 24px rgba(15,23,42,0.24);
}


/* =========================================================
   SELECTED MOVIE
   ========================================================= */

.selected-heading {
    font-size: 35px;
    font-weight: 900;

    color: #ffffff;

    margin-top: 38px;
    margin-bottom: 8px;
}

.selected-description {
    color: #b9c5d8;

    font-size: 14px;

    margin-bottom: 15px;
}


/* =========================================================
   GENRE CHIPS
   ========================================================= */

.genre-chip {
    display: inline-block;

    padding: 7px 14px;
    margin: 4px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(129,140,248,0.27),
            rgba(34,211,238,0.15)
        );

    border:
        1px solid rgba(165,180,252,0.38);

    color: #e0e7ff;

    font-size: 12px;
    font-weight: 750;

    box-shadow:
        0 4px 14px rgba(30,41,59,0.20);
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(43,55,87,0.94),
            rgba(35,46,79,0.94)
        );

    border:
        1px solid rgba(165,180,252,0.18);

    border-radius: 17px;

    padding: 16px;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.25);
}

div[data-testid="stMetricLabel"] {
    color: #b8c4d8 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 900;
}


/* =========================================================
   OVERVIEW
   ========================================================= */

.overview-title {
    font-size: 18px;
    font-weight: 800;

    color: #ffffff;

    margin-top: 23px;
    margin-bottom: 8px;
}

.overview-text {
    color: #c5cfdd;

    font-size: 15px;
    line-height: 1.8;

    max-width: 950px;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 30px;
    font-weight: 900;

    color: #ffffff;

    margin-top: 48px;
    margin-bottom: 24px;

    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: "";

    height: 2px;
    flex: 1;

    margin-left: 12px;

    background:
        linear-gradient(
            90deg,
            rgba(165,180,252,0.65),
            rgba(34,211,238,0.25),
            transparent
        );
}


/* =========================================================
   MOVIE INFORMATION
   ========================================================= */

.movie-name {
    font-size: 16px;
    font-weight: 800;

    color: #ffffff;

    line-height: 1.4;

    min-height: 44px;

    margin-top: 10px;
}

.movie-info {
    color: #b9c5d8;

    font-size: 13px;

    margin-top: 6px;
}


/* =========================================================
   MATCH BADGE
   ========================================================= */

.match-score {
    display: inline-block;

    margin-top: 9px;

    padding: 5px 11px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.20),
            rgba(34,211,238,0.15)
        );

    border:
        1px solid rgba(52,211,153,0.30);

    color: #86efac;

    font-size: 12px;
    font-weight: 800;
}


/* =========================================================
   BUTTONS
   ========================================================= */

div.stButton > button {
    width: 100%;

    min-height: 46px;

    border-radius: 13px;

    border:
        1px solid rgba(196,181,253,0.34);

    background:
        linear-gradient(
            135deg,
            #6366f1,
            #7c3aed,
            #a855f7
        );

    color: #ffffff;

    font-size: 14px;
    font-weight: 800;

    box-shadow:
        0 8px 22px rgba(99,102,241,0.25);

    transition:
        transform 0.20s ease,
        box-shadow 0.20s ease;
}

div.stButton > button:hover {
    transform: translateY(-3px);

    box-shadow:
        0 14px 32px rgba(99,102,241,0.42);
}


/* =========================================================
   EXPANDER
   ========================================================= */

div[data-testid="stExpander"] {
    border:
        1px solid rgba(165,180,252,0.18);

    border-radius: 13px;

    background:
        rgba(35,46,79,0.58);
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    border-color:
        rgba(165,180,252,0.13) !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer-box {
    text-align: center;

    margin-top: 75px;

    padding-top: 28px;

    border-top:
        1px solid rgba(165,180,252,0.14);

    color: #8290a5;

    font-size: 13px;

    line-height: 1.9;
}

.footer-brand {
    color: #c4b5fd;

    font-size: 16px;
    font-weight: 850;
}


/* =========================================================
   STREAMLIT CLEANUP
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .hero-title {
        font-size: 37px;
    }

    .hero-box {
        padding: 43px 20px;
    }

    .selected-heading {
        font-size: 29px;
    }

    .section-title {
        font-size: 25px;
    }

}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero-box">
<div class="hero-icon">🎬</div>
<div class="hero-title"> Movie Recommendation System </div>
<div class="hero-subtitle">Discover your next favorite movie with intelligent recommendations</div>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# SEARCH
# ============================================================

st.markdown(
"""
<div class="search-heading">🔍 Find a Movie</div>
""",
unsafe_allow_html=True
)


movie_titles = sorted(
    recommendation_df[
        "title"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


current_movie = st.session_state.selected_movie


if current_movie in movie_titles:

    default_index = movie_titles.index(
        current_movie
    )

else:

    default_index = None


selected_movie = st.selectbox(
    "Choose a movie",
    movie_titles,
    index=default_index,
    placeholder="Type or select a movie..."
)


# ============================================================
# UPDATE SELECTED MOVIE
# ============================================================

if selected_movie:

    if (
        st.session_state.selected_movie
        != selected_movie
    ):

        st.session_state.selected_movie = (
            selected_movie
        )

        st.session_state.show_recommendations = False


current_movie = (
    st.session_state.selected_movie
)


# ============================================================
# SELECTED MOVIE
# ============================================================

if current_movie:

    selected_data = recommendation_df[
        recommendation_df["title"] == current_movie
    ]

    if not selected_data.empty:

        movie = selected_data.iloc[0]

        st.markdown(
            """
<div class="selected-heading">
🎞️ Selected Movie
</div>
""",
            unsafe_allow_html=True
        )

        poster_col, info_col = st.columns(
            [1, 2.7],
            gap="large"
        )


        # ----------------------------------------------------
        # POSTER
        # ----------------------------------------------------

        with poster_col:

            poster_path = get_poster_path(
                movie["id"]
            )

            if poster_path:

                st.image(
                    poster_path,
                    width=280
                )

            else:

                st.info(
                    "Poster not available"
                )


        # ----------------------------------------------------
        # MOVIE INFORMATION
        # ----------------------------------------------------

        with info_col:

            st.markdown(
                f"""
<div class="selected-heading">
{safe_text(movie["title"])}
</div>

<div class="selected-description">
Your selected movie
</div>
""",
                unsafe_allow_html=True
            )


            # Genres
            genres = get_genres(movie)

            if genres:

                genre_html = ""

                for genre in genres:

                    genre_html += (
                        f'<span class="genre-chip">'
                        f'{safe_text(genre)}'
                        f'</span>'
                    )

                st.markdown(
                    genre_html,
                    unsafe_allow_html=True
                )


            st.write("")


            # Metrics
            rating = float(
                movie["vote_average"]
            )

            year_text = get_year(movie)

            language = get_language(movie)


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "⭐ Rating",
                    f"{rating:.1f}/10"
                )


            with col2:

                st.metric(
                    "📅 Release",
                    year_text
                )


            with col3:

                st.metric(
                    "🌍 Language",
                    language
                )


            # Overview
            if "overview_clean" in movie.index:

                overview = movie[
                    "overview_clean"
                ]

                if (
                    pd.notna(overview)
                    and
                    str(overview).strip()
                ):

                    st.markdown(
                        """
<div class="overview-title">
📝 Overview
</div>
""",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
<div class="overview-text">
{safe_text(overview)}
</div>
""",
                        unsafe_allow_html=True
                    )


# ============================================================
# RECOMMEND BUTTON
# ============================================================

st.write("")

recommend_clicked = st.button(
    "✨ Discover Similar Movies"
)


if recommend_clicked:

    if current_movie:

        st.session_state.show_recommendations = True

    else:

        st.warning(
            "Please select a movie first."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if (
    current_movie
    and
    st.session_state.show_recommendations
):

    recommendations = recommend_movies(
        current_movie,
        top_n=10
    )


    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if recommendations.empty:

        st.warning(
            "No poster-available recommendations "
            "were found for this movie."
        )


    else:

        st.markdown(
            """
<div class="section-title">
🍿 Recommended For You
</div>
""",
            unsafe_allow_html=True
        )


        source_rows = recommendation_df[
            recommendation_df["title"] == current_movie
        ]

        source_movie = source_rows.iloc[0]


        # ----------------------------------------------------
        # 5 MOVIES PER ROW
        # ----------------------------------------------------

        for start in range(
            0,
            len(recommendations),
            5
        ):

            row = recommendations.iloc[
                start:start + 5
            ]

            cols = st.columns(
                5,
                gap="medium"
            )


            for position, (
                col,
                (_, rec)
            ) in enumerate(
                zip(
                    cols,
                    row.iterrows()
                )
            ):

                with col:

                    # Poster
                    poster_path = get_poster_path(
                        rec["id"]
                    )

                    if poster_path:

                        st.image(
                            poster_path,
                            width=200
                        )


                    # Movie title
                    st.markdown(
                        f"""
<div class="movie-name">
{safe_text(rec["title"])}
</div>
""",
                        unsafe_allow_html=True
                    )


                    # Rating + Year
                    st.markdown(
                        f"""
<div class="movie-info">
⭐ {float(rec["vote_average"]):.1f}/10
&nbsp;&nbsp;•&nbsp;&nbsp;
📅 {get_year(rec)}
</div>
""",
                        unsafe_allow_html=True
                    )


                    # Match
                    similarity = float(
                        rec["similarity_score"]
                    )

                    match_percentage = (
                        similarity * 100
                    )

                    match_percentage = max(
                        0,
                        min(
                            100,
                            match_percentage
                        )
                    )


                    st.markdown(
                        f"""
<div class="match-score">
🎯 {match_percentage:.0f}% Match
</div>
""",
                        unsafe_allow_html=True
                    )


                    # Explanation
                    reasons = explain_recommendation(
                        source_movie,
                        rec
                    )


                    with st.expander(
                        "💡 Why?"
                    ):

                        for reason in reasons:

                            st.write(
                                f"• {reason}"
                            )


                    # View & Recommend
                    button_key = (
                        f"view_movie_"
                        f"{int(rec['id'])}_"
                        f"{start}_"
                        f"{position}"
                    )


                    if st.button(
                        "🎬 View & Recommend",
                        key=button_key
                    ):

                        st.session_state.selected_movie = (
                            rec["title"]
                        )

                        st.session_state.show_recommendations = (
                            True
                        )

                        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
<div class="footer-box">

<div class="footer-brand">
🎬 Movie Recommendation System
</div>

Intelligent Content-Based Movie Discovery

<br>

"Built with Python • Machine Learning • Streamlit"

</div>
""",
unsafe_allow_html=True
)