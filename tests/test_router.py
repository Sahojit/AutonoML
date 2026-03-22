import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.query_router import (
    RouteType,
    ResultCache,
    classify_route,
    _is_simple_query,
)


class TestIsSimpleQuery:

    def test_what_is_javascript(self):
        assert _is_simple_query("what is javascript") is True

    def test_what_are_transformers(self):
        assert _is_simple_query("what are transformers") is True

    def test_who_is_alan_turing(self):
        assert _is_simple_query("who is alan turing") is True

    def test_explain_keyword(self):
        assert _is_simple_query("explain recursion") is True

    def test_define_keyword(self):
        assert _is_simple_query("define entropy") is True

    def test_how_does_gradient_descent_work(self):
        assert _is_simple_query("how does gradient descent work") is True

    def test_very_short_query(self):
        assert _is_simple_query("sorting algorithms overview") is True

    def test_long_query_is_not_simple(self):
        q = "can you explain in detail how the transformer architecture works"
        assert _is_simple_query(q) is False

    def test_realtime_keyword_disqualifies(self):
        assert _is_simple_query("what is the current bitcoin price") is False

    def test_realtime_keyword_today(self):
        assert _is_simple_query("what is today weather") is False

    def test_empty_string_does_not_crash(self):
        result = _is_simple_query("")
        assert isinstance(result, bool)

    def test_only_spaces_does_not_crash(self):
        result = _is_simple_query("   ")
        assert isinstance(result, bool)

    def test_single_word_is_simple(self):
        assert _is_simple_query("python") is True

    def test_only_stopwords_does_not_crash(self):
        result = _is_simple_query("the a is")
        assert isinstance(result, bool)


class TestClassifyRouteDirect:

    def test_what_is_javascript(self):
        assert classify_route("what is javascript") == RouteType.DIRECT

    def test_explain_recursion(self):
        assert classify_route("explain recursion") == RouteType.DIRECT

    def test_define_overfitting(self):
        assert classify_route("define overfitting") == RouteType.DIRECT

    def test_simple_query_ignores_memory_context(self):
        assert classify_route(
            "what is python", has_memory_context=True, has_conversation=True
        ) == RouteType.DIRECT

    def test_no_flags_generic_query(self):
        assert classify_route("tell me about the solar system") == RouteType.DIRECT

    def test_empty_query_does_not_crash(self):
        result = classify_route("")
        assert isinstance(result, RouteType)

    def test_only_stopwords_does_not_crash(self):
        result = classify_route("the and or")
        assert isinstance(result, RouteType)

    def test_mixed_case_handled(self):
        assert classify_route("What IS JavaScript") == RouteType.DIRECT


class TestClassifyRouteTool:

    def test_latest_nvidia_news(self):
        assert classify_route("latest nvidia news") == RouteType.TOOL

    def test_current_bitcoin_price(self):
        assert classify_route("current bitcoin price") == RouteType.TOOL

    def test_todays_weather(self):
        assert classify_route("what is todays weather in london") == RouteType.TOOL

    def test_stock_price(self):
        assert classify_route("apple stock price today") == RouteType.TOOL

    def test_crypto_exchange_rate(self):
        assert classify_route("ethereum exchange rate now") == RouteType.TOOL

    def test_trending_topics(self):
        assert classify_route("what is trending on twitter right now") == RouteType.TOOL

    def test_breaking_news(self):
        assert classify_route("breaking news in tech") == RouteType.TOOL

    def test_realtime_overrides_simple_start(self):
        assert classify_route("what is the current stock market index") == RouteType.TOOL


class TestClassifyRoutePipeline:

    def test_train_random_forest(self):
        assert classify_route("train a random forest on the iris dataset") == RouteType.PIPELINE

    def test_analyse_dataset(self):
        assert classify_route("analyse the wine dataset") == RouteType.PIPELINE

    def test_compare_ml_models(self):
        assert classify_route("compare ml models on the diabetes dataset") == RouteType.PIPELINE

    def test_build_classifier(self):
        assert classify_route("build a classifier using xgboost") == RouteType.PIPELINE

    def test_write_code(self):
        assert classify_route("write code to scrape a website") == RouteType.PIPELINE

    def test_deep_learning(self):
        assert classify_route("implement a deep learning model for image classification") == RouteType.PIPELINE

    def test_auto_experiment(self):
        assert classify_route("auto-experiment on the iris sklearn dataset") == RouteType.PIPELINE

    def test_generate_report(self):
        assert classify_route("generate a report from the dataset") == RouteType.PIPELINE


class TestClassifyRouteRag:

    def test_followup_improve_it(self):
        assert classify_route("improve it", has_conversation=True, has_memory_context=True) == RouteType.RAG

    def test_followup_fix_it(self):
        assert classify_route("fix it", has_conversation=True, has_memory_context=True) == RouteType.RAG

    def test_followup_keyword_previous(self):
        assert classify_route("what did we do in the previous session") == RouteType.RAG

    def test_followup_keyword_earlier(self):
        assert classify_route("show me the results from earlier") == RouteType.RAG

    def test_followup_keyword_last_time(self):
        assert classify_route("do the same as last time") == RouteType.RAG

    def test_both_flags_true_no_keyword(self):
        assert classify_route(
            "can you recap what we discussed",
            has_memory_context=True,
            has_conversation=True,
        ) == RouteType.RAG

    def test_only_memory_context_no_convo_is_direct(self):
        assert classify_route(
            "how does backpropagation work",
            has_memory_context=True,
            has_conversation=False,
        ) == RouteType.DIRECT

    def test_only_conversation_no_memory_is_direct(self):
        assert classify_route(
            "how does backpropagation work",
            has_memory_context=False,
            has_conversation=True,
        ) == RouteType.DIRECT


class TestResultCache:

    def test_miss_on_empty_cache(self):
        cache = ResultCache(ttl=300)
        assert cache.get("what is python", "") is None

    def test_hit_after_set(self):
        cache = ResultCache(ttl=300)
        payload = {"final_answer": "Python is a language.", "route": "direct"}
        cache.set("what is python", "", payload)
        result = cache.get("what is python", "")
        assert result is not None
        assert result["final_answer"] == "Python is a language."

    def test_cache_key_is_normalised(self):
        cache = ResultCache(ttl=300)
        payload = {"final_answer": "answer"}
        cache.set("  What IS Python  ", "", payload)
        assert cache.get("what is python", "") is not None

    def test_miss_after_ttl_expiry(self):
        cache = ResultCache(ttl=0.05)
        payload = {"final_answer": "answer"}
        cache.set("query", "", payload)
        time.sleep(0.1)
        assert cache.get("query", "") is None

    def test_different_ctx_is_different_key(self):
        cache = ResultCache(ttl=300)
        payload = {"final_answer": "answer"}
        cache.set("what is python", "ctx-A", payload)
        assert cache.get("what is python", "ctx-B") is None

    def test_evict_expired_removes_stale_entries(self):
        cache = ResultCache(ttl=0.05)
        cache.set("q1", "", {"final_answer": "a1"})
        cache.set("q2", "", {"final_answer": "a2"})
        time.sleep(0.1)
        evicted = cache.evict_expired()
        assert evicted == 2
        assert cache.size == 0

    def test_clear_empties_cache(self):
        cache = ResultCache(ttl=300)
        cache.set("q1", "", {"final_answer": "a1"})
        cache.set("q2", "", {"final_answer": "a2"})
        cache.clear()
        assert cache.size == 0

    def test_size_property(self):
        cache = ResultCache(ttl=300)
        assert cache.size == 0
        cache.set("q1", "", {"k": "v"})
        assert cache.size == 1
        cache.set("q2", "", {"k": "v"})
        assert cache.size == 2
