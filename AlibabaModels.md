# Alibaba Model Studio - Coding Plan Models

This document provides detailed specifications for models available through Alibaba Cloud Model Studio's Coding Plan.

## Table of Contents

- [Qwen Models](#qwen-models)
  - [Qwen3.5 Plus](#qwen35-plus)
  - [Qwen3 Max 2026-01-23](#qwen3-max-2026-01-23)
  - [Qwen3 Coder Next](#qwen3-coder-next)
  - [Qwen3 Coder Plus](#qwen3-coder-plus)
- [GLM Models](#glm-models)
  - [GLM-5](#glm-5)
  - [GLM-4.7](#glm-47)
- [Kimi Models](#kimi-models)
  - [Kimi K2.5](#kimi-k25)
- [MiniMax Models](#minimax-models)
  - [MiniMax M2.5](#minimax-m25)

---

## Qwen Models

### Qwen3.5 Plus

A balanced model with strong reasoning and multimodal capabilities.

| Property | Value |
|----------|-------|
| **ID** | qwen3.5-plus |
| **Family** | qwen |
| **Context Window** | 1,000,000 tokens |
| **Max Output** | 65,536 tokens |
| **Max CoT** | 81,920 tokens |
| **Reasoning** | Yes (enabled by default) |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text, image, video |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 256K | $0.40 | $2.40 |
| 256K - 1M | $0.50 | $3.00 |

**Notes**: Thinking mode is enabled by default.

---

### Qwen3 Max 2026-01-23

The most powerful Qwen model for complex tasks, with thinking mode support.

| Property | Value |
|----------|-------|
| **ID** | qwen3-max-2026-01-23 |
| **Family** | qwen |
| **Context Window** | 262,144 tokens |
| **Max Output** | 32,768 tokens |
| **Max CoT** | 81,920 tokens |
| **Reasoning** | Yes |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 32K | $1.20 | $6.00 |
| 32K - 128K | $2.40 | $12.00 |
| 128K - 252K | $3.00 | $15.00 |

**Notes**: Thinking mode integrates web search, web extractor, and code interpreter tools.

---

### Qwen3 Coder Next

Next-generation coding model with large context support.

| Property | Value |
|----------|-------|
| **ID** | qwen3-coder-next |
| **Family** | qwen |
| **Context Window** | 262,144 tokens |
| **Max Output** | 65,536 tokens |
| **Reasoning** | No |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | Yes |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 32K | $0.30 | $1.50 |
| 32K - 128K | $0.50 | $2.50 |
| 128K - 256K | $0.80 | $4.00 |

---

### Qwen3 Coder Plus

Advanced coding model with agent capabilities and maximum context.

| Property | Value |
|----------|-------|
| **ID** | qwen3-coder-plus |
| **Family** | qwen |
| **Context Window** | 1,000,000 tokens |
| **Max Output** | 65,536 tokens |
| **Reasoning** | No |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | Yes |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 32K | $1.00 | $5.00 |
| 32K - 128K | $1.80 | $9.00 |
| 128K - 256K | $3.00 | $15.00 |
| 256K - 1M | $6.00 | $60.00 |

**Notes**: Supports context cache (implicit: 20%, explicit: 10% pricing).

---

## GLM Models

Hybrid reasoning models from Zhipu AI, designed specifically for agents.

**Note**: GLM models are Chinese mainland only (Beijing region).

### GLM-5

Latest flagship model from Zhipu AI with SOTA coding performance.

| Property | Value |
|----------|-------|
| **ID** | glm-5 |
| **Family** | glm |
| **Context Window** | 202,752 tokens |
| **Max Output** | 16,384 tokens |
| **Max CoT** | 32,768 tokens |
| **Reasoning** | Yes |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 32K | $0.573 | $2.58 |
| 32K - 166K | $0.86 | $3.154 |

**Notes**: Charges same rate for thinking and non-thinking modes.

---

### GLM-4.7

Previous generation GLM model with strong reasoning capabilities.

| Property | Value |
|----------|-------|
| **ID** | glm-4.7 |
| **Family** | glm |
| **Context Window** | 169,984 tokens |
| **Max Output** | ~16,384 tokens |
| **Reasoning** | Yes |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Cost** (per 1M tokens):
| Input Tokens | Input Cost | Output Cost |
|--------------|------------|-------------|
| 0 - 32K | $0.431 | $2.007 |
| 32K - 166K | $0.574 | $2.294 |

**Notes**: Charges same rate for thinking and non-thinking modes.

---

## Kimi Models

Large language models developed by Moonshot AI, optimized for coding and tool calling.

**Note**: Kimi models are Chinese mainland only (Beijing region).

### Kimi K2.5

Advanced multimodal model with strong coding and visual understanding capabilities.

| Property | Value |
|----------|-------|
| **ID** | kimi-k2.5 |
| **Family** | kimi |
| **Context Window** | 262,144 tokens |
| **Max Output** | 32,768 tokens |
| **Max CoT** | 32,768 tokens |
| **Reasoning** | Yes |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text, image |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Cost** (per 1M tokens):
| Mode | Input Cost | Output Cost |
|------|------------|-------------|
| Thinking | $0.574 | $3.011 |
| Non-thinking | $0.574 | $3.011 |

**Notes**: Supports both thinking and non-thinking modes.

---

## MiniMax Models

Text models from MiniMax with strong code generation capabilities.

### MiniMax M2.5

Latest text model from MiniMax with peak performance for complex tasks.

| Property | Value |
|----------|-------|
| **ID** | MiniMax-M2.5 |
| **Family** | minimax |
| **Context Window** | 200,000 tokens |
| **Max Output** | 128,000 tokens |
| **Reasoning** | Yes |
| **Tool Calling** | Yes |
| **Modalities (Input)** | text |
| **Modalities (Output)** | text |
| **Open Weights** | No |

**Features**:
- Optimized for code generation and refactoring
- Polyglot code mastery
- Precision code refactoring
- Advanced reasoning
- Real-time streaming

---

## Deployment Regions

| Model Family | Region |
|--------------|--------|
| Qwen (most) | International, Global, US, Chinese Mainland |
| GLM | Chinese Mainland only (Beijing) |
| Kimi | Chinese Mainland only (Beijing) |
| MiniMax | Varies by configuration |

---

## Additional Notes

- **Context Cache**: Some models support context caching which reduces costs for repeated context
- **Thinking Mode**: Enables reasoning capabilities with optional budget tokens
- **Tool Calling**: Support for function calling and autonomous agent capabilities
- **Open Weights**: Models available for self-hosting

For the most up-to-date information, see [Alibaba Cloud Model Studio Documentation](https://www.alibabacloud.com/help/en/model-studio/models).
