import i18n from 'i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import { defaultLanguage, resources, supportedLanguages } from './resources'
import { syncManifestLanguage } from './manifest'

i18n.use(LanguageDetector).init({
  defaultNS: 'common',
  detection: {
    order: ['cookie', 'navigator'],
    caches: ['cookie'],
    cookieOptions: {
      path: '/',
      sameSite: 'lax',
      secure: typeof location !== 'undefined' && location.protocol === 'https:',
      maxAge: 60 * 60 * 24 * 365,
    },
  },
  fallbackLng: defaultLanguage,
  interpolation: {
    escapeValue: false,
  },
  ns: Object.keys(resources[defaultLanguage]),
  resources,
  returnNull: false,
  supportedLngs: supportedLanguages,
})

export default i18n

function syncDocumentLanguage() {
  if (typeof document !== 'undefined') {
    document.documentElement.lang = i18n.resolvedLanguage ?? defaultLanguage
    syncManifestLanguage(document.documentElement.lang)
  }
}

i18n.on('languageChanged', syncDocumentLanguage)
syncDocumentLanguage()
