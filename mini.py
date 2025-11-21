import cv2
import os
import numpy as np

# Threshold for detection
thres = 0.5

# Pricing dictionary for detected items
item_prices = {
    "1.person":0,
    "item12 bottle":200,
    "item33 wine glass":350,
    "item64 banana":40,
    "item20 apple":30,
    "item10 cake":100,
    "item23 carrot":32,
    "item24 oranges":54,
    "bicycle": 5000,
    "item65 car toy":100,
    "2 boat": 3000000,
    "item10 cup":20,
    "item56 traffic light": 5000
    # Add other classes and prices as needed
}

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load class names
classFile = 'coco.names'
if not os.path.exists(classFile):
    print("Error: coco.names file not found.")
    exit()

with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(f"Loaded {len(classNames)} class names.")

# Load model
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14(2).pbtxt"

if not (os.path.exists(weightsPath) and os.path.exists(configPath)):
    print("Error: Model files not found in the current directory.")
    exit()

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

print("Model loaded successfully.")

# Initialize detected items list
detected_items = []

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(f"Detected classes: {classIds}")

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 1 <= classId <= len(classNames):  # Ensure classId is valid
                label = classNames[classId - 1]
                detected_items.append(label)

                cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                cv2.putText(img, label.upper(), (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img, f"{round(confidence * 100, 2)}%", (box[0], box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Create a white panel for the invoice
    invoice_panel = np.ones((480, 400, 3), dtype=np.uint8) * 255  # White panel

    # Add title "Invoice" on the panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(invoice_panel, "Invoice", (10, 90), font, 1, (0, 0, 255), 2)

    # Draw the heading
    cv2.putText(invoice_panel, "Self Checkout System", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # Prepare the invoice items
    invoice = {}
    total_cost = 0
    item_counts = {}  # Dictionary to track item counts

    for item in detected_items:
        if item in item_prices:
            # Increment the count of the detected item
            item_counts[item] = item_counts.get(item, 0) + 1
            # Calculate the cost for the item based on its count
            invoice[item] = item_counts[item] * item_prices[item]

    # Calculate total cost
    total_cost = sum(invoice.values())

    # Draw the invoice items with counts
    y_offset = 120
    for item, cost in invoice.items():
        count = item_counts[item]  # Get the count of the item
        cv2.putText(invoice_panel, f"{item} (x{count}): Rs. {cost}", (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 0), 1)
        y_offset += 30

    # Draw the total cost
    cv2.putText(invoice_panel, f"Total: Rs. {total_cost}", (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Ensure both images have the same height
    if img.shape[0] != invoice_panel.shape[0]:
        invoice_panel = cv2.resize(invoice_panel, (invoice_panel.shape[1], img.shape[0]))

    # Combine the images horizontally
    combined_img = np.hstack((img, invoice_panel))

    cv2.imshow("Output", combined_img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()