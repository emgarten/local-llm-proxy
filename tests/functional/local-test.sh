#!/bin/bash

# add a provider to to your config named 'test'

curl -X POST "http://localhost:8000/provider/test/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?"
            }
        ],
        "max_completion_tokens": 2048,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }'