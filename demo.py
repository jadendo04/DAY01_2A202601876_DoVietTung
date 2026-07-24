from template import format_comparison_table

fake_results = [
    {
        "prompt": "Việt Nam có bao nhiêu tỉnh?",
        "gpt4o_answer": "Việt Nam hiện có 63 tỉnh thành trên cả nước, bao gồm...",
        "mini_answer": "63 tỉnh thành.",
        "gpt4o_time": 2.481,
        "mini_time": 1.023,
        "gpt4o_cost": 0.0001,
    },
]
print(format_comparison_table(fake_results))