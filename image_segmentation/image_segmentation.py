# Image Segmentation
# Resim Bölütleme

import cv2

# load pre-trained model.
# önceden eğitilmiş modeli yükleyin.
model = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")

# initialize video stream.
# video akışını başlat.
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video stream.
    # video akışından bir kare oku.
    ret, frame = cap.read()

    # Resize the frame to the model input size.
    # Çerçeveyi model giriş boyutuna göre yeniden boyutlandırın.
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # set the input for the model
    # model için girişi ayarlayın
    model.setInput(blob)

    # make a forward pass through the model
    # modelden ileri geçiş yapın
    output = model.forward()

    # loop over the detections and draw the segmentation mask on the frame.
    # tespitler üzerinde döngü yapın ve segmentasyon maskesini çerçeveye çizin.
    for i in range(output.shape[1]):
        confidence = output[0, i, 2]
        if confidence > 0.5:
            mask = output[0, i, 1]
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask > 0.5).astype("uint8") * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, mask)

    # show the frame
    # çerçeveyi göster
    cv2.imshow("Frame", frame)

    # exit if the user presses the 'q' key
    # kullanıcı 'q' tuşuna basarsa çıkar
    if cv2.waitKey(1) == ord('q'):
        break

# let's finish the process.
# işlemi bitirelim.
cap.release()
cv2.destroyAllWindows()