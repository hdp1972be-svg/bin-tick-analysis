import ccxt
import paho.mqtt.client as mqtt
import asyncio
import json

# Configuration
EXCHANGE_NAME = 'binance'  # Replace with your desired exchange (e.g., 'binance', 'kraken')
MQTT_BROKER = 'localhost'  # Replace with your MQTT broker address
MQTT_PORT = 1883           # Default MQTT port
MQTT_TOPIC = 'crypto/tickers'  # MQTT topic to publish ticker data

# Initialize the exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Initialize MQTT client
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    if rc == 0:
        print("MQTT connection successful!")
    else:
        print(f"Failed to connect to MQTT broker, error code: {rc}")

# Connect to MQTT broker
mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

async def fetch_tickers():
    while True:
        try:
            # Fetch all tickers from the exchange
            tickers = await exchange.fetch_tickers()
            
            # Publish each ticker to the MQTT topic
            for symbol, ticker in tickers.items():
                payload = json.dumps(ticker)
                mqtt_client.publish(MQTT_TOPIC, payload)
                print(f"Published {symbol}: {payload}")
            
            # Wait for a short period before fetching again
            await asyncio.sleep(10)  # Adjust the delay as needed

        except Exception as e:
            print(f"Error fetching tickers: {e}")
            await asyncio.sleep(10)  # Retry after a delay

async def main():
    # Start the MQTT loop
    mqtt_client.loop_start()

    # Start fetching tickers
    await fetch_tickers()

# Run the program
if __name__ == "__main__":
    asyncio.run(main())