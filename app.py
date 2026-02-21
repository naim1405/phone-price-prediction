import gradio as gr
import pandas as pd
import pickle

# model_path = "mobile_price_predictor.pkl"
model_path = "./model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)


def perform_inference(
    battery_power, blue, clock_speed, dual_sim, fc, four_g,
    int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
    px_width, ram, sc_h, sc_w, talk_time, three_g,
    touch_screen, wifi
):

    input_data = pd.DataFrame([{
        "battery_power": battery_power,
        "clock_speed": clock_speed,
        "fc": fc,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": int(three_g),
        "touch_screen": int(touch_screen),
        "dual_sim": int(dual_sim),
        "four_g": int(four_g),
        "wifi": int(wifi),
        "blue": int(blue),
    }])

    prediction = model.predict(input_data)[0]

    return f"Predicted Price Range: {prediction}"


app = gr.Interface(
    fn=perform_inference,
    inputs=[
        gr.Number(label="Battery Power (mAh)"),
        gr.Number(label="Clock Speed (GHz)"),
        gr.Number(label="Front Camera (MP)"),
        gr.Number(label="Internal Memory (GB)"),
        gr.Number(label="Mobile Depth (cm)"),
        gr.Number(label="Mobile Weight (g)"),
        gr.Number(label="Number of Cores"),
        gr.Number(label="Primary Camera (MP)"),
        gr.Number(label="Pixel Height"),
        gr.Number(label="Pixel Width"),
        gr.Number(label="RAM (MB)"),
        gr.Number(label="Screen Height (cm)"),
        gr.Number(label="Screen Width (cm)"),
        gr.Number(label="Talk Time (hours)"),
        gr.Checkbox(label="Touch Screen"),
        gr.Checkbox(label="Dual SIM"),
        gr.Checkbox(label="4G"),
        gr.Checkbox(label="3G"),
        gr.Checkbox(label="WiFi"),
        gr.Checkbox(label="Bluetooth"),
    ],
    outputs=gr.Textbox(label="Predicted Price Range"),
    title="Phone Price Predictor"
)

if __name__ == "__main__":
    app.launch()
