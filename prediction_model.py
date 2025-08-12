import math
import numpy as np
import streamlit as st

class PredictionModel:
    def poisson(self, k, lam):
        """Calculate Poisson probability"""
        try:
            return (math.exp(-lam) * lam**k) / math.factorial(k)
        except (OverflowError, ValueError):
            return 0

    def odds_to_prob(self, o1, x, o2):
        """Convert 1X2 odds to probabilities"""
        try:
            p1 = 1 / o1
            px = 1 / x
            p2 = 1 / o2
            total = p1 + px + p2
            return p1 / total, px / total, p2 / total
        except:
            return 0.33, 0.33, 0.33

    def match_probs(self, lambda1, lambda2):
        """Calculate H/D/A probabilities from xG values"""
        max_goals = 10
        p_home, p_draw, p_away = 0, 0, 0
        try:
            for i in range(max_goals):
                for j in range(max_goals):
                    p = self.poisson(i, lambda1) * self.poisson(j, lambda2)
                    if i > j:
                        p_home += p
                    elif i == j:
                        p_draw += p
                    else:
                        p_away += p
        except:
            p_home, p_draw, p_away = 0.33, 0.33, 0.33
        return p_home, p_draw, p_away

    def estimate_xg_from_odds(self, odds_home, odds_draw, odds_away):
        """Convert 1X2 odds to xG values using optimization"""
        try:
            p_target_home, p_target_draw, p_target_away = self.odds_to_prob(odds_home, odds_draw, odds_away)
            min_error = float('inf')
            best_lambda1, best_lambda2 = 1.0, 1.0
            for l1 in [round(x * 0.1, 1) for x in range(5, 35)]:
                for l2 in [round(x * 0.1, 1) for x in range(5, 35)]:
                    ph, pd, pa = self.match_probs(l1, l2)
                    error = (ph - p_target_home)**2 + (pd - p_target_draw)**2 + (pa - p_target_away)**2
                    if error < min_error:
                        min_error = error
                        best_lambda1 = l1
                        best_lambda2 = l2
            return round(best_lambda1, 2), round(best_lambda2, 2), min_error
        except:
            return 1.5, 1.5, 0.1

    def calculate_score_probabilities(self, lambda_home, lambda_away):
        """Calculate all score probabilities using Poisson distribution"""
        try:
            prob_home = {}
            prob_away = {}
            for i in range(10):
                prob_home[i] = self.poisson(i, lambda_home)
                prob_away[i] = self.poisson(i, lambda_away)
            prob_result = {}
            for i in range(10):
                prob_result[i] = {}
                for j in range(10):
                    prob_result[i][j] = prob_home[i] * prob_away[j]
            return prob_result
        except:
            prob_result = {}
            for i in range(10):
                prob_result[i] = {}
                for j in range(10):
                    prob_result[i][j] = 0.01
            return prob_result

    def get_most_probable_scores(self, prob_result, num_scores):
        """METHOD 1: Get most probable scores by probability order"""
        try:
            all_scores = [(i, j, prob_result[i][j]) for i in range(10) for j in range(10)]
            most_probable = sorted(all_scores, key=lambda x: x[2], reverse=True)[:num_scores]
            return most_probable
        except:
            return [(1, 1, 0.1)] * min(num_scores, 5)

    def predict_hda_from_scores(self, scores):
        """Predict H/D/A from score list using probability weighting"""
        try:
            if not scores:
                return 'D', 33.33, {'H': 33.33, 'D': 33.33, 'A': 33.33}
            total_prob = sum(prob for _, _, prob in scores)
            h_prob = sum(prob for home, away, prob in scores if home > away)
            d_prob = sum(prob for home, away, prob in scores if home == away)
            a_prob = sum(prob for home, away, prob in scores if home < away)
            if total_prob > 0:
                h_prob /= total_prob
                d_prob /= total_prob
                a_prob /= total_prob
            if h_prob > d_prob and h_prob > a_prob:
                prediction = 'H'
                confidence = h_prob * 100
            elif d_prob > h_prob and d_prob > a_prob:
                prediction = 'D'
                confidence = d_prob * 100
            else:
                prediction = 'A'
                confidence = a_prob * 100
            return prediction, confidence, {'H': h_prob * 100, 'D': d_prob * 100, 'A': a_prob * 100}
        except:
            return 'D', 33.33, {'H': 33.33, 'D': 33.33, 'A': 33.33}

    def predict_ou_from_scores(self, scores):
        """Predict Over/Under 2.5 from score list"""
        try:
            if not scores:
                return 'UNDER 2.5', 50.0, {'OVER': 50.0, 'UNDER': 50.0}
            total_prob = sum(prob for _, _, prob in scores)
            over_prob = sum(prob for home, away, prob in scores if home + away > 2.5)
            under_prob = sum(prob for home, away, prob in scores if home + away <= 2.5)
            if total_prob > 0:
                over_prob /= total_prob
                under_prob /= total_prob
            if over_prob > under_prob:
                prediction = 'OVER 2.5'
                confidence = over_prob * 100
            else:
                prediction = 'UNDER 2.5'
                confidence = under_prob * 100
            return prediction, confidence, {'OVER': over_prob * 100, 'UNDER': under_prob * 100}
        except:
            return 'UNDER 2.5', 50.0, {'OVER': 50.0, 'UNDER': 50.0}

    def calculate_weighted_avg(self, scores):
        """Calculate weighted average for pattern analysis"""
        try:
            if not scores:
                return 0
            total_prob = sum(prob for _, _, prob in scores)
            if total_prob > 0:
                weighted_avg = sum((home + away) * prob for home, away, prob in scores) / total_prob
            else:
                weighted_avg = sum(home + away for home, away, _ in scores) / len(scores)
            return weighted_avg
        except:
            return 2.0

    def predict(self, odds_home, odds_draw, odds_away):
        """
        Makes predictions based on the optimal parameters found in the analysis.
        """
        lambda_home, lambda_away, _ = self.estimate_xg_from_odds(odds_home, odds_draw, odds_away)
        prob_result = self.calculate_score_probabilities(lambda_home, lambda_away)

        # --- Primary Predictions ---
        # H/D/A Prediction (Method 1, 11 scores)
        scores_hda = self.get_most_probable_scores(prob_result, 11)
        hda_pred, hda_conf, _ = self.predict_hda_from_scores(scores_hda)

        # O/U Prediction (Method 1, 17 scores)
        scores_ou = self.get_most_probable_scores(prob_result, 17)
        ou_pred, ou_conf, _ = self.predict_ou_from_scores(scores_ou)

        # --- Special Pattern Checks ---
        # High Confidence H/D/A (10 scores)
        scores_high_conf = self.get_most_probable_scores(prob_result, 10)
        hda_pred_hc, hda_conf_hc, _ = self.predict_hda_from_scores(scores_high_conf)
        high_confidence_hda_signal = hda_conf_hc >= 60

        # High AVG Pattern O/U (8 scores)
        scores_high_avg = self.get_most_probable_scores(prob_result, 8)
        weighted_avg = self.calculate_weighted_avg(scores_high_avg)
        high_avg_pattern_signal = weighted_avg >= 2.5
        ou_pred_hap, ou_conf_hap, _ = self.predict_ou_from_scores(scores_high_avg)


        return {
            'primary_hda': {
                'prediction': hda_pred,
                'confidence': hda_conf,
                'details': "Using Method 1 with 11 scores"
            },
            'primary_ou': {
                'prediction': ou_pred,
                'confidence': ou_conf,
                'details': "Using Method 1 with 17 scores"
            },
            'special_patterns': {
                'high_confidence_hda': {
                    'triggered': high_confidence_hda_signal,
                    'prediction': hda_pred_hc if high_confidence_hda_signal else "N/A",
                    'confidence': hda_conf_hc if high_confidence_hda_signal else "N/A",
                    'details': "Signal for H/D/A when confidence is >= 60% using 10 scores"
                },
                'high_avg_pattern_ou': {
                    'triggered': high_avg_pattern_signal,
                    'prediction': "OVER 2.5" if high_avg_pattern_signal else "N/A",
                    'confidence': ou_conf_hap if high_avg_pattern_signal else "N/A",
                    'details': "Signal for OVER 2.5 when weighted goal average is >= 2.5 using 8 scores"
                }
            }
        }

def main():
    st.set_page_config(page_title="⚽ Match Predictor", layout="wide")
    st.title("🎯 Football Match Outcome Predictor")
    st.markdown("Enter the 1X2 odds for a match to get predictions based on an optimized Poisson model.")

    model = PredictionModel()

    with st.sidebar:
        st.header("Match Odds")
        odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.5, step=0.1)
        odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.2, step=0.1)
        odds_away = st.number_input("Away Win Odds", min_value=1.01, value=2.8, step=0.1)

        predict_button = st.button("🔮 Get Prediction", use_container_width=True)

    if predict_button:
        with st.spinner('🧠 Analyzing odds and running predictions...'):
            predictions = model.predict(odds_home, odds_draw, odds_away)

            st.subheader("📈 Primary Predictions")
            col1, col2 = st.columns(2)
            with col1:
                hda = predictions['primary_hda']
                st.metric(
                    label="Match Outcome (H/D/A)",
                    value=hda['prediction'],
                    delta=f"{hda['confidence']:.1f}% Confidence"
                )
            with col2:
                ou = predictions['primary_ou']
                st.metric(
                    label="Goal Total (Over/Under 2.5)",
                    value=ou['prediction'],
                    delta=f"{ou['confidence']:.1f}% Confidence"
                )

            st.subheader("⚡️ High-Accuracy Pattern Signals")
            st.info("These signals identify specific scenarios where the model has historically shown higher accuracy.")

            col1, col2 = st.columns(2)
            with col1:
                hda_sp = predictions['special_patterns']['high_confidence_hda']
                if hda_sp['triggered']:
                    st.success(f"✅ High-Confidence H/D/A Signal: **{hda_sp['prediction']}**")
                    st.write(f"Confidence: **{hda_sp['confidence']:.1f}%**")
                    st.write(f"_{hda_sp['details']}_")
                else:
                    st.write("No High-Confidence H/D/A signal.")

            with col2:
                ou_sp = predictions['special_patterns']['high_avg_pattern_ou']
                if ou_sp['triggered']:
                    st.success(f"✅ High-AVG-Pattern O/U Signal: **{ou_sp['prediction']}**")
                    st.write(f"Confidence: **{ou_sp['confidence']:.1f}%**")
                    st.write(f"_{ou_sp['details']}_")
                else:
                    st.write("No High-AVG-Pattern O/U signal.")

if __name__ == '__main__':
    main()
