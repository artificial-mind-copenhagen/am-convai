# am-convai

Thin flask wrapper around Hugging Face's ConvAi2 - so it can be used as a fallback.

Be aware that the docker image takes forever to build (tokenizing the dataset is a pain).

## Usage

### `/ready`

Sending a HTTP GET to `/ready` responds with either:

```json
{"success": True}
```

or

```json
{"error": "Model isn't ready"}
```

depending on whether the service is finishe initializing yet.


### `/conversation`

Send a HTTP POST request containing something that resembles the following:

```json
{
  "history": [
    "None shall pass.",
    "What?",
    "None shall pass.",
    "I have no quarrel with you, good Sir knight, but I must cross this bridge.",
    "Then you shall die."
  ],
  "query": "I command you as King of the Britons to stand aside!"
}
```

and the response will contain a JSON object that looks like this:

```json
{
    "response": "i'll be the best king"
}
```
