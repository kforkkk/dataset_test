from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv("/hsk/.env")

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("CHATANYWHERE_KEY"),
    base_url=os.getenv("CHATANYWHERE_URL")
    
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def generate_video_description(caption, image_path):
    sys = f"""

Write a one-sentence description of a short video that begins with the provided image and smoothly transitions into a scene of a {caption}, highlighting how elements in the image undergo changes or movement over time. 
Keep the description simple, concise and short, focusing only on essential changes and actions without altering unnecessary details. 
Avoid mentioning elements that do not contribute to the main change needed, and focus the description on the main transitions. 
Do not add objects that are not in the original image or described in the final scene. 
The camera should remain static unless movement is absolutely necessary. 
Ensure all transitions happen within a few second duration without mentioning the length or using the word "video".


"""
    encoded_image = encode_image(image_path)

    brige = encode_image("/hsk/F2F/assets/bridge.png")
    brige_input = "make the brige collapse into the water"
    brige_output = "Bricks at the center of the bridge begin to fall downward, triggering a progressive collapse that spreads symmetrically outward until the entire structure disintegrates."

    cloud = encode_image("/hsk/F2F/assets/cloud.png")
    cloud_input = "make it a sky without cloud"
    cloud_output = "Clouds drift steadily leftward until they completely vanish from the frame, leaving an empty sky."

    coffee = encode_image("/hsk/F2F/assets/coffee.png")
    coffee_input = "make the american coffee into a cup of tea"
    coffee_output = "The black coffee in the cup gradually recedes and is replaced as a teapot pours amber tea until the cup fills, then the teapot withdraws from view."

    cokkie = encode_image("/hsk/F2F/assets/cokkie.png")
    cokkie_input = "every cookie split into two pieces"
    cokkie_output = "A knife enters the frame, slices vertically through a stack of cookies, then lifts and withdraws from view."

    cow = encode_image("/hsk/F2F/assets/cow.png")
    cow_input = "make the cow into a tiger"
    cow_output = "The cow's features gradually morph—ears sharpening, stripes emerging, and body reshaping—until it fully transforms into a tiger while remaining stationary."

    door = encode_image("/hsk/F2F/assets/door.png")
    door_input = "make the door open"
    door_output = "The door gradually swings open, revealing the space beyond while the camera remains fixed."

    flower = encode_image("/hsk/F2F/assets/flower.png")
    flower_input = "make the flower into a rose"
    flower_output = "The painted flower gradually withers, then revives with intensifying color until it fully transforms into a vibrant crimson rose while remaining in place."

    sun = encode_image("/hsk/F2F/assets/sun.png")
    sun_input = "turn it to night"
    sun_output = "The sun slowly fades into the background, becoming a dark, bluish-gray disc that slowly disappears from view."

    tree = encode_image("/hsk/F2F/assets/tree.png")
    tree_input = "what it looks like in summer"
    tree_output = "Snow on the branches melts and drips to the ground where it evaporates, while green buds sprout and the tree deepens to lush green—all within a fixed frame."



    messages=[
        {
            "role": "user",
            "content": [
                {"type":"text","text":sys},
                {"type":"text","text":"here is the provided image"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{encoded_image}"},
                },
                {"type":"text","text":"here is example one"},
                {"type":"text","text":f"input:{brige_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{brige}"},
                },
                {"type":"text","text":f"output:{brige_output}"},
                
                {"type":"text","text":"here is example two"},
                {"type":"text","text":f"input:{cloud_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{cloud}"},
                },
                {"type":"text","text":f"output:{cloud_output}"},
                
                {"type":"text","text":"here is example three"},
                {"type":"text","text":f"input:{coffee_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{coffee}"},
                },
                {"type":"text","text":f"output:{coffee_output}"},
                
                {"type":"text","text":"here is example four"},
                {"type":"text","text":f"input:{cokkie_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{cokkie}"},
                },
                {"type":"text","text":f"output:{cokkie_output}"},
                
                {"type":"text","text":"here is example five"},
                {"type":"text","text":f"input:{cow_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{cow}"},
                },
                {"type":"text","text":f"output:{cow_output}"},
                
                {"type":"text","text":"here is example six"},
                {"type":"text","text":f"input:{door_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{door}"},
                },
                {"type":"text","text":f"output:{door_output}"},
                
                {"type":"text","text":"here is example seven"},
                {"type":"text","text":f"input:{flower_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{flower}"},
                },
                {"type":"text","text":f"output:{flower_output}"},
                
                {"type":"text","text":"here is example eight"},
                {"type":"text","text":f"input:{sun_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{sun}"},
                },
                {"type":"text","text":f"output:{sun_output}"},
                
                {"type":"text","text":"here is example nine"},
                {"type":"text","text":f"input:{tree_input}"},
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{tree}"},
                },
                {"type":"text","text":f"output:{tree_output}"}

            ],
        },
    ]

    completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,

        )
    if hasattr(completion.choices[0], 'message'):
    # 获取完整消息内容
        full_message = completion.choices[0].message

        answer_content = full_message.content or ""
        print("enhanced_content:", answer_content)
        return answer_content


# 示例调用
#result = generate_video_description("make the white flower into a black one", "/hsk/F2F/assets/try.png")