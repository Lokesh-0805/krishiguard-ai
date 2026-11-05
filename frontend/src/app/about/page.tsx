"use client";

import { AlertTriangle, BookOpen, HeartHandshake, Lock, Scale } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useLang } from "@/lib/i18n";

const COPY = {
  en: {
    intro: "KrishiGuard AI is a portfolio and research project exploring how deep learning can support — not replace — crop-health decisions by farmers and extension workers in India.",
    blocks: [
      ["What it does", "Identifies 38 crop–condition classes from a leaf photo, reports calibrated confidence, explains its focus with Grad-CAM and shows advisory content in English and Hindi with cited sources."],
      ["What it does not do", "It cannot diagnose crops or diseases outside its 38 classes, multiple simultaneous problems, nutrient deficiencies or pest damage it was not trained on. Accuracy on field photos is much lower than on lab photos — see the performance page."],
      ["Safety", "Treatment advice is shown only for confident results. No dosages are given; chemical options always point to the registered product label and local experts (KVK / agriculture officer)."],
      ["Privacy", "Uploaded photos are processed in memory and not stored. Only anonymous metadata (prediction, confidence, image hash, timing) is logged. Feedback is stored as unverified and never used as ground truth without expert review."],
    ],
    credits: "Data credits",
  },
  hi: {
    intro: "KrishiGuard AI एक शोध और पोर्टफ़ोलियो परियोजना है, जो यह दिखाती है कि डीप लर्निंग भारत के किसानों और कृषि विस्तार कर्मियों के फसल-स्वास्थ्य निर्णयों में मदद कैसे कर सकती है — उनकी जगह नहीं ले सकती।",
    blocks: [
      ["यह क्या करता है", "पत्ती की फ़ोटो से 38 फसल-रोग स्थितियाँ पहचानता है, अपना भरोसा बताता है, Grad-CAM से दिखाता है कि उसने कहाँ देखा, और स्रोतों के साथ हिंदी-अंग्रेज़ी में सलाह देता है।"],
      ["यह क्या नहीं करता", "यह इन 38 स्थितियों के बाहर की फसलें या रोग, एक साथ कई समस्याएँ, पोषक तत्वों की कमी या कीट नुकसान नहीं पहचान सकता। खेत की फ़ोटो पर इसकी सटीकता लैब फ़ोटो से काफ़ी कम है — प्रदर्शन पेज देखें।"],
      ["सुरक्षा", "उपचार की सलाह केवल भरोसेमंद नतीजों पर दिखाई जाती है। कोई मात्रा (डोज़) नहीं बताई जाती; रासायनिक उपायों के लिए हमेशा पंजीकृत उत्पाद लेबल और स्थानीय विशेषज्ञ (KVK / कृषि अधिकारी) की सलाह लें।"],
      ["निजता", "अपलोड की गई फ़ोटो सहेजी नहीं जाती। केवल गुमनाम जानकारी (परिणाम, भरोसा, समय) दर्ज होती है। आपकी प्रतिक्रिया 'असत्यापित' रहती है और विशेषज्ञ जाँच के बिना उपयोग नहीं होती।"],
    ],
    credits: "डेटा आभार",
  },
} as const;
const ICONS = [BookOpen, AlertTriangle, Scale, Lock];

export default function AboutPage() {
  const { t, lang } = useLang();
  const c = COPY[lang];
  return (
    <div className="mx-auto max-w-5xl space-y-10 px-4 py-10 sm:px-6 sm:py-14">
      <div className="space-y-3">
        <h1 className="text-3xl font-semibold sm:text-4xl">{t("about_title")}</h1>
        <p className="max-w-3xl text-lg text-muted-foreground">{c.intro}</p>
      </div>
      <div className="grid gap-5 sm:grid-cols-2">
        {c.blocks.map(([title, body], i) => {
          const Icon = ICONS[i];
          return (
            <Card key={title}>
              <CardHeader><CardTitle className="flex items-center gap-2"><Icon className="size-5 text-primary" />{title}</CardTitle></CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">{body}</CardContent>
            </Card>
          );
        })}
      </div>
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><HeartHandshake className="size-5 text-primary" />{c.credits}</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>Mohanty S.P., Hughes D.P., Salathé M. (2016). Using deep learning for image-based plant disease detection. <i>Frontiers in Plant Science</i> 7:1419 — PlantVillage dataset (CC BY-SA 3.0).</p>
          <p>Singh D. et al. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. CODS-COMAD 2020 (CC BY 4.0).</p>
          <p>Tomato Leaf Disease Classification Dataset in Pakistan, Mendeley Data 3mbnb82mxd (CC BY 4.0).</p>
          <p>Advisory content is compiled from ICAR institutes, state agricultural universities, government portals and university extension services; each entry lists its sources.</p>
        </CardContent>
      </Card>
    </div>
  );
}
