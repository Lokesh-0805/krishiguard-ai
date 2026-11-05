"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Lang } from "./api";

const dict = {
  en: {
    nav_home: "Home", nav_detect: "Diagnose", nav_library: "Disease library", nav_performance: "Model performance",
    nav_method: "Methodology", nav_about: "About", cta_diagnose: "Diagnose a leaf", lang_label: "हिंदी",
    hero_kicker: "AI crop health assistant",
    hero_title: "Spot crop disease early. Act with confidence.",
    hero_sub: "Upload or capture a photo of a single leaf. KrishiGuard identifies the crop and disease, tells you how sure it is, shows where it looked, and gives bilingual advice from agricultural sources.",
    hero_secondary: "See how accurate it is",
    detect_title: "Diagnose a leaf",
    detect_sub: "Use a clear, well-lit photo of one leaf, filling most of the frame. Your photo is analysed and not stored.",
    drop_title: "Drag & drop a leaf photo", drop_or: "or", browse: "Choose file", camera: "Use camera",
    drop_hint: "JPEG, PNG or WebP · up to 8 MB",
    replace: "Replace photo", remove: "Remove", analyse: "Analyse leaf", analysing: "Analysing…",
    capture: "Capture", cancel: "Cancel", camera_error: "Camera unavailable. Check browser permission or upload a photo instead.",
    tips_title: "For a reliable result",
    tip1: "One leaf, in focus, filling most of the photo", tip2: "Daylight or even light — avoid harsh shadows",
    tip3: "Show the affected side of the leaf", tip4: "Supported crops only (see list)",
    supported_crops: "Supported crops",
    err_type: "Please choose a JPEG, PNG or WebP image.", err_size: "This image is larger than 8 MB.",
    result_title: "Diagnosis", crop: "Crop", disease: "Condition", confidence: "Confidence", crop_conf: "Crop confidence",
    healthy: "Healthy", diseased: "Diseased", status_confident: "Confident result", status_uncertain: "Uncertain result",
    status_unrecognized: "Not recognised",
    other_candidates: "Other possibilities", symptoms: "Symptoms", causes: "Causes", prevention: "Prevention",
    organic: "Organic options", chemical: "Chemical options", overview: "Overview", treatment: "Treatment",
    sources: "Sources", no_chemical: "No chemical treatment is recommended for this condition.",
    explain_title: "Where the model looked (Grad-CAM)", gradcam_note: "Highlights the image regions that most influenced the predicted class. It is not proof that the diagnosis is correct.", original: "Model input", heatmap: "Heatmap", overlay: "Overlay",
    download_report: "Download report", print: "Print / Save PDF", new_diagnosis: "New diagnosis",
    no_result: "No diagnosis yet.", go_detect: "Go to the diagnosis page",
    quality_blurry: "The photo looks blurry — results may be unreliable.",
    quality_lowres: "Low-resolution photo — move closer to the leaf.",
    feedback_title: "Was this helpful?", fb_correct: "Diagnosis correct", fb_incorrect: "Diagnosis wrong",
    fb_helpful: "Advice helpful", fb_comment: "Optional comment (e.g. what the expert said)", fb_send: "Send feedback",
    fb_thanks: "Thank you. Feedback is stored as unverified and reviewed before any use.",
    advisory_hidden: "Treatment advice is shown only for confident results. Review the possibilities below and consult your local Krishi Vigyan Kendra (KVK).",
    view_advisory: "View advisory",
    library_title: "Disease library", library_sub: "Every condition the model can recognise, with symptoms, prevention and treatment options from cited agricultural sources.",
    search: "Search crop or disease…", all_crops: "All crops", close: "Close",
    perf_title: "Model performance", perf_sub: "Measured on held-out data that was never used for training or model selection.",
    method_title: "Methodology", about_title: "About KrishiGuard AI",
    footer_disclaimer: "KrishiGuard AI is a decision-support tool, not a substitute for an agricultural expert. Always confirm a diagnosis and follow registered product labels.",
    loading: "Loading…", retry: "Retry", backend_down: "The analysis service is not reachable right now.",
  },
  hi: {
    nav_home: "होम", nav_detect: "जाँच करें", nav_library: "रोग जानकारी", nav_performance: "मॉडल प्रदर्शन",
    nav_method: "कार्यप्रणाली", nav_about: "परिचय", cta_diagnose: "पत्ती की जाँच करें", lang_label: "English",
    hero_kicker: "AI फसल स्वास्थ्य सहायक",
    hero_title: "फसल रोग को जल्दी पहचानें। भरोसे से कदम उठाएँ।",
    hero_sub: "एक पत्ती की फ़ोटो अपलोड करें या कैमरे से लें। KrishiGuard फसल और रोग पहचानता है, बताता है कि वह कितना निश्चित है, दिखाता है कि उसने फ़ोटो में कहाँ देखा, और कृषि स्रोतों पर आधारित हिंदी-अंग्रेज़ी सलाह देता है।",
    hero_secondary: "इसकी सटीकता देखें",
    detect_title: "पत्ती की जाँच करें",
    detect_sub: "एक पत्ती की साफ़, अच्छी रोशनी वाली फ़ोटो लें जो फ़्रेम का ज़्यादातर हिस्सा भरे। आपकी फ़ोटो की जाँच होती है, उसे सहेजा नहीं जाता।",
    drop_title: "पत्ती की फ़ोटो यहाँ खींचकर छोड़ें", drop_or: "या", browse: "फ़ाइल चुनें", camera: "कैमरा चलाएँ",
    drop_hint: "JPEG, PNG या WebP · 8 MB तक",
    replace: "फ़ोटो बदलें", remove: "हटाएँ", analyse: "जाँच शुरू करें", analysing: "जाँच हो रही है…",
    capture: "फ़ोटो लें", cancel: "रद्द करें", camera_error: "कैमरा उपलब्ध नहीं है। ब्राउज़र की अनुमति जाँचें या फ़ोटो अपलोड करें।",
    tips_title: "सही नतीजे के लिए",
    tip1: "एक ही पत्ती, साफ़ फ़ोकस में, फ़ोटो का ज़्यादातर हिस्सा भरे", tip2: "दिन की रोशनी या बराबर रोशनी — तेज़ छाया से बचें",
    tip3: "पत्ती का रोग वाला हिस्सा दिखाएँ", tip4: "केवल समर्थित फसलें (सूची देखें)",
    supported_crops: "समर्थित फसलें",
    err_type: "कृपया JPEG, PNG या WebP फ़ोटो चुनें।", err_size: "यह फ़ोटो 8 MB से बड़ी है।",
    result_title: "जाँच का नतीजा", crop: "फसल", disease: "स्थिति", confidence: "भरोसा", crop_conf: "फसल पहचान का भरोसा",
    healthy: "स्वस्थ", diseased: "रोगग्रस्त", status_confident: "भरोसेमंद नतीजा", status_uncertain: "अनिश्चित नतीजा",
    status_unrecognized: "पहचान नहीं हुई",
    other_candidates: "अन्य संभावनाएँ", symptoms: "लक्षण", causes: "कारण", prevention: "बचाव",
    organic: "जैविक उपाय", chemical: "रासायनिक उपाय", overview: "सारांश", treatment: "उपचार",
    sources: "स्रोत", no_chemical: "इस स्थिति के लिए कोई रासायनिक उपचार सुझाया नहीं गया है।",
    explain_title: "मॉडल ने कहाँ देखा (Grad-CAM)", gradcam_note: "फ़ोटो के वे हिस्से रंगीन दिखते हैं जिनका मॉडल के फ़ैसले पर सबसे ज़्यादा असर पड़ा। यह इस बात का सबूत नहीं है कि पहचान सही है।", original: "मॉडल को दी गई फ़ोटो", heatmap: "हीटमैप", overlay: "ओवरले",
    download_report: "रिपोर्ट डाउनलोड करें", print: "प्रिंट / PDF सहेजें", new_diagnosis: "नई जाँच",
    no_result: "अभी कोई जाँच नहीं हुई।", go_detect: "जाँच पेज पर जाएँ",
    quality_blurry: "फ़ोटो धुंधली लग रही है — नतीजा गलत हो सकता है।",
    quality_lowres: "फ़ोटो छोटी है — पत्ती के पास जाकर फ़ोटो लें।",
    feedback_title: "क्या यह मददगार था?", fb_correct: "पहचान सही है", fb_incorrect: "पहचान गलत है",
    fb_helpful: "सलाह उपयोगी है", fb_comment: "टिप्पणी (वैकल्पिक, जैसे विशेषज्ञ ने क्या कहा)", fb_send: "प्रतिक्रिया भेजें",
    fb_thanks: "धन्यवाद। आपकी प्रतिक्रिया 'असत्यापित' रूप में सहेजी गई है और उपयोग से पहले जाँची जाएगी।",
    advisory_hidden: "उपचार की सलाह केवल भरोसेमंद नतीजे पर दिखाई जाती है। नीचे दी गई संभावनाएँ देखें और अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से सलाह लें।",
    view_advisory: "सलाह देखें",
    library_title: "रोग जानकारी", library_sub: "मॉडल जिन स्थितियों को पहचान सकता है, उनके लक्षण, बचाव और उपचार — प्रमाणित कृषि स्रोतों के साथ।",
    search: "फसल या रोग खोजें…", all_crops: "सभी फसलें", close: "बंद करें",
    perf_title: "मॉडल प्रदर्शन", perf_sub: "ऐसे डेटा पर मापा गया जो न प्रशिक्षण में और न मॉडल चुनने में इस्तेमाल हुआ।",
    method_title: "कार्यप्रणाली", about_title: "KrishiGuard AI के बारे में",
    footer_disclaimer: "KrishiGuard AI निर्णय में मदद करने वाला उपकरण है, कृषि विशेषज्ञ का विकल्प नहीं। हमेशा जाँच की पुष्टि करें और पंजीकृत उत्पाद के लेबल का पालन करें।",
    loading: "लोड हो रहा है…", retry: "फिर कोशिश करें", backend_down: "जाँच सेवा अभी उपलब्ध नहीं है।",
  },
} as const;

export type TKey = keyof (typeof dict)["en"];

interface Ctx { lang: Lang; setLang: (l: Lang) => void; t: (k: TKey) => string }
const LangContext = createContext<Ctx | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("en");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("kg-lang");
      // eslint-disable-next-line react-hooks/set-state-in-effect -- hydrate persisted preference after mount
      if (saved === "hi" || saved === "en") setLangState(saved);
    } catch { /* storage unavailable */ }
  }, []);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try { localStorage.setItem("kg-lang", l); } catch { /* ignore */ }
  }, []);
  const value = useMemo(() => ({ lang, setLang, t: (k: TKey) => dict[lang][k] }), [lang, setLang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const c = useContext(LangContext);
  if (!c) throw new Error("useLang must be used inside LanguageProvider");
  return c;
}
