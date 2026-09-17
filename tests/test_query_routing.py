from store_assistant.query_routing import infer_source_type


def test_routes_operational_inventory_question_to_delivery_logs() -> None:
    assert infer_source_type("Do we have oat milk cartons?") == "delivery_log"


def test_routes_recipe_and_contract_questions() -> None:
    assert infer_source_type("How do I make overnight oats?") == "recipe"
    assert infer_source_type("When does the supplier contract renew?") == "supplier_contract"


def test_leaves_unknown_or_ambiguous_questions_unrestricted() -> None:
    assert infer_source_type("Tell me about oat milk") is None
    assert infer_source_type("Does the contract cover recipe ingredients?") is None
