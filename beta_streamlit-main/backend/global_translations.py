"""
Global translation system for LAISA application.
Provides centralized text translations for all Streamlit pages.
"""

import streamlit as st
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global translations dictionary
GLOBAL_TRANSLATIONS = {
    'en': {
        # Common UI elements
        'upload_document': 'Upload Document',
        'drag_drop_files': 'Drop your documents here',
        'processing': 'Processing',
        'loading': 'Loading',
        'upload_completed': 'Upload completed!',
        'upload_failed': 'Upload failed',
        'access_granted': 'Access Granted',
        'access_denied': 'Access Denied',
        'invalid_pin': 'Invalid PIN',
        'admin_only': 'ADMIN ONLY',
        'restricted_access': 'Restricted Access',
        'enter_admin_pin': 'Enter Admin PIN',
        'contact_help': 'Contact & Government Housing Offices',
        'find_local_assistance': 'Find Local Housing Assistance',
        'office_details': 'Office Details',
        'address': 'Address',
        'phone': 'Phone',
        'hours': 'Hours',
        'services_available': 'Services Available',
        'emergency_housing': 'Emergency Housing',
        'important_note': 'Important Note',
        'document_analysis': 'Document analysis is for informational purposes only - Not legal advice',
        'need_more_help': 'Need More Help?',
        'common_questions': 'Common Questions',
        'questions_about_document': 'Questions about this document?',
        'go_to_chat': 'Go to the Chat page to ask specific questions',
        'upload_to_kb': 'Upload this document to the Document Upload page to add it to LAISA\'s knowledge base',
        'consult_legal': 'Consult with a legal professional for specific legal advice',
        'document_summary': 'Document Summary',
        'document_type': 'Document Type',
        'fields_to_fill': 'Fields to Fill',
        'section_explanations': 'Section Explanations',
        'system_status': 'System Status',
        'all_systems_operational': 'All systems operational',
        'system_check_failed': 'System check failed',
        'check_system_status': 'Check System Status',
        'admin_information': 'Admin Information',
        'document_processing': 'Document Processing',
        'supported_formats': 'Supported Formats',
        'security_note': 'Security Note',
        'language_support': 'Language Support',
        'active_language': 'Active Language',
        'detected_language': 'Detected Language',
        'confidence': 'Confidence',
        'source': 'Source',
        'translation_active': 'Translation: Active',
        'translation_inactive': 'Translation: Inactive (low confidence)',
        'method': 'Method',
        'script': 'Script',
        'indicators': 'Indicators',
        'statistical': 'Statistical',
        'ai': 'AI',
        'alternative_candidates': 'Alternative candidates',
        'current_session': 'Current Session',
        'messages': 'Messages',
        'language_english_default': 'Language: English (Default)',
        'housing_assistant': 'Housing Assistant',
        'multilingual_ai_assistant': 'Multilingual AI assistant for housing-related questions',
        'reset_chat': 'Reset Chat',
        'topics_help_with': 'Topics I Can Help With',
        'tenant_rights': 'Tenant rights & responsibilities',
        'landlord_obligations': 'Landlord obligations',
        'rental_agreements': 'Rental agreements & leases',
        'housing_regulations': 'Housing regulations',
        'eviction_processes': 'Eviction processes',
        'property_management': 'Property management laws',
        'housing_discrimination': 'Housing discrimination',
        'under_construction': 'Under Construction',
        'documentation': 'Documentation',
        'contact': 'Contact',
        'document_help': 'Document Help',
        'data_upload': 'Data Upload',
        'upload_analysis': 'Upload a document to get analysis and assistance',
        'pdf_only_supported': 'Only PDF files are supported. Convert other formats to PDF first.',
        'failed_read_pdf': 'Failed to read PDF',
        'try_different_method': 'Please try a different method to convert your document to PDF format',
        'analyzing_document': 'Analyzing document',
        'loading_document': 'Loading document',
        'upload_pdf_to_start': 'Upload a PDF document to get started'
    },
    'es': {
        # Common UI elements
        'upload_document': 'Subir Documento',
        'drag_drop_files': 'Arrastra tus documentos aquí',
        'processing': 'Procesando',
        'loading': 'Cargando',
        'upload_completed': '¡Carga completada!',
        'upload_failed': 'Error en la carga',
        'access_granted': 'Acceso Concedido',
        'access_denied': 'Acceso Denegado',
        'invalid_pin': 'PIN Inválido',
        'admin_only': 'SOLO ADMINISTRADOR',
        'restricted_access': 'Acceso Restringido',
        'enter_admin_pin': 'Ingrese PIN de Administrador',
        'contact_help': 'Contacto y Oficinas Gubernamentales de Vivienda',
        'find_local_assistance': 'Encuentra Asistencia Local de Vivienda',
        'office_details': 'Detalles de la Oficina',
        'address': 'Dirección',
        'phone': 'Teléfono',
        'hours': 'Horarios',
        'services_available': 'Servicios Disponibles',
        'emergency_housing': 'Vivienda de Emergencia',
        'important_note': 'Nota Importante',
        'document_analysis': 'El análisis de documentos es solo para fines informativos - No es asesoramiento legal',
        'need_more_help': '¿Necesitas Más Ayuda?',
        'common_questions': 'Preguntas Comunes',
        'questions_about_document': '¿Preguntas sobre este documento?',
        'go_to_chat': 'Ve a la página de Chat para hacer preguntas específicas',
        'upload_to_kb': 'Sube este documento a la página de Carga de Documentos para agregarlo a la base de conocimientos de LAISA',
        'consult_legal': 'Consulta con un profesional legal para asesoramiento legal específico',
        'document_summary': 'Resumen del Documento',
        'document_type': 'Tipo de Documento',
        'fields_to_fill': 'Campos a Completar',
        'section_explanations': 'Explicaciones de Secciones',
        'system_status': 'Estado del Sistema',
        'all_systems_operational': 'Todos los sistemas operativos',
        'system_check_failed': 'Verificación del sistema falló',
        'check_system_status': 'Verificar Estado del Sistema',
        'admin_information': 'Información del Administrador',
        'document_processing': 'Procesamiento de Documentos',
        'supported_formats': 'Formatos Soportados',
        'security_note': 'Nota de Seguridad',
        'language_support': 'Soporte de Idiomas',
        'active_language': 'Idioma Activo',
        'detected_language': 'Idioma Detectado',
        'confidence': 'Confianza',
        'source': 'Fuente',
        'translation_active': 'Traducción: Activa',
        'translation_inactive': 'Traducción: Inactiva (baja confianza)',
        'method': 'Método',
        'script': 'Escritura',
        'indicators': 'Indicadores',
        'statistical': 'Estadístico',
        'ai': 'IA',
        'alternative_candidates': 'Candidatos alternativos',
        'current_session': 'Sesión Actual',
        'messages': 'Mensajes',
        'language_english_default': 'Idioma: Inglés (Predeterminado)',
        'housing_assistant': 'Asistente de Vivienda',
        'multilingual_ai_assistant': 'Asistente de IA multilingüe para preguntas relacionadas con vivienda',
        'reset_chat': 'Reiniciar Chat',
        'topics_help_with': 'Temas con los que Puedo Ayudar',
        'tenant_rights': 'Derechos y responsabilidades del inquilino',
        'landlord_obligations': 'Obligaciones del propietario',
        'rental_agreements': 'Acuerdos y contratos de alquiler',
        'housing_regulations': 'Regulaciones de vivienda',
        'eviction_processes': 'Procesos de desalojo',
        'property_management': 'Leyes de gestión de propiedades',
        'housing_discrimination': 'Discriminación de vivienda',
        'under_construction': 'En Construcción',
        'documentation': 'Documentación',
        'contact': 'Contacto',
        'document_help': 'Ayuda con Documentos',
        'data_upload': 'Carga de Datos',
        'upload_analysis': 'Sube un documento para obtener análisis y asistencia',
        'pdf_only_supported': 'Solo se admiten archivos PDF. Convierte otros formatos a PDF primero.',
        'failed_read_pdf': 'Error al leer PDF',
        'try_different_method': 'Por favor, intenta un método diferente para convertir tu documento a formato PDF',
        'analyzing_document': 'Analizando documento',
        'loading_document': 'Cargando documento',
        'upload_pdf_to_start': 'Sube un documento PDF para comenzar'
    },
    'fr': {
        # Common UI elements
        'upload_document': 'Télécharger le Document',
        'drag_drop_files': 'Déposez vos documents ici',
        'processing': 'Traitement',
        'loading': 'Chargement',
        'upload_completed': 'Téléchargement terminé !',
        'upload_failed': 'Échec du téléchargement',
        'access_granted': 'Accès Accordé',
        'access_denied': 'Accès Refusé',
        'invalid_pin': 'PIN Invalide',
        'admin_only': 'ADMINISTRATEUR SEULEMENT',
        'restricted_access': 'Accès Restreint',
        'enter_admin_pin': 'Entrez le PIN Administrateur',
        'contact_help': 'Contact et Bureaux Gouvernementaux du Logement',
        'find_local_assistance': 'Trouvez une Assistance Locale au Logement',
        'office_details': 'Détails du Bureau',
        'address': 'Adresse',
        'phone': 'Téléphone',
        'hours': 'Heures',
        'services_available': 'Services Disponibles',
        'emergency_housing': 'Logement d\'Urgence',
        'important_note': 'Note Importante',
        'document_analysis': 'L\'analyse de document est à des fins informatives uniquement - Pas de conseil juridique',
        'need_more_help': 'Besoin de Plus d\'Aide ?',
        'common_questions': 'Questions Courantes',
        'questions_about_document': 'Questions sur ce document ?',
        'go_to_chat': 'Allez à la page Chat pour poser des questions spécifiques',
        'upload_to_kb': 'Téléchargez ce document sur la page de Téléchargement de Documents pour l\'ajouter à la base de connaissances de LAISA',
        'consult_legal': 'Consultez un professionnel juridique pour des conseils juridiques spécifiques',
        'document_summary': 'Résumé du Document',
        'document_type': 'Type de Document',
        'fields_to_fill': 'Champs à Remplir',
        'section_explanations': 'Explications des Sections',
        'system_status': 'État du Système',
        'all_systems_operational': 'Tous les systèmes opérationnels',
        'system_check_failed': 'Vérification du système échouée',
        'check_system_status': 'Vérifier l\'État du Système',
        'admin_information': 'Informations Administrateur',
        'document_processing': 'Traitement des Documents',
        'supported_formats': 'Formats Supportés',
        'security_note': 'Note de Sécurité',
        'language_support': 'Support Linguistique',
        'active_language': 'Langue Active',
        'detected_language': 'Langue Détectée',
        'confidence': 'Confiance',
        'source': 'Source',
        'translation_active': 'Traduction : Active',
        'translation_inactive': 'Traduction : Inactive (faible confiance)',
        'method': 'Méthode',
        'script': 'Écriture',
        'indicators': 'Indicateurs',
        'statistical': 'Statistique',
        'ai': 'IA',
        'alternative_candidates': 'Candidats alternatifs',
        'current_session': 'Session Actuelle',
        'messages': 'Messages',
        'language_english_default': 'Langue : Anglais (Par défaut)',
        'housing_assistant': 'Assistant Logement',
        'multilingual_ai_assistant': 'Assistant IA multilingue pour les questions liées au logement',
        'reset_chat': 'Réinitialiser le Chat',
        'topics_help_with': 'Sujets avec Lesquels Je Peux Aider',
        'tenant_rights': 'Droits et responsabilités des locataires',
        'landlord_obligations': 'Obligations du propriétaire',
        'rental_agreements': 'Accords et baux de location',
        'housing_regulations': 'Réglementations du logement',
        'eviction_processes': 'Processus d\'expulsion',
        'property_management': 'Lois de gestion immobilière',
        'housing_discrimination': 'Discrimination au logement',
        'under_construction': 'En Construction',
        'documentation': 'Documentation',
        'contact': 'Contact',
        'document_help': 'Aide aux Documents',
        'data_upload': 'Téléchargement de Données',
        'upload_analysis': 'Téléchargez un document pour obtenir une analyse et une assistance',
        'pdf_only_supported': 'Seuls les fichiers PDF sont pris en charge. Convertissez d\'abord les autres formats en PDF.',
        'failed_read_pdf': 'Échec de la lecture du PDF',
        'try_different_method': 'Veuillez essayer une méthode différente pour convertir votre document au format PDF',
        'analyzing_document': 'Analyse du document',
        'loading_document': 'Chargement du document',
        'upload_pdf_to_start': 'Téléchargez un document PDF pour commencer'
    },
    'de': {
        # Common UI elements
        'upload_document': 'Dokument Hochladen',
        'drag_drop_files': 'Ziehen Sie Ihre Dokumente hierher',
        'processing': 'Verarbeitung',
        'loading': 'Laden',
        'upload_completed': 'Upload abgeschlossen!',
        'upload_failed': 'Upload fehlgeschlagen',
        'access_granted': 'Zugang Gewährt',
        'access_denied': 'Zugang Verweigert',
        'invalid_pin': 'Ungültige PIN',
        'admin_only': 'NUR ADMINISTRATOR',
        'restricted_access': 'Eingeschränkter Zugang',
        'enter_admin_pin': 'Administrator-PIN eingeben',
        'contact_help': 'Kontakt und Regierungswohnungsämter',
        'find_local_assistance': 'Lokale Wohnungsberatung Finden',
        'office_details': 'Bürodetails',
        'address': 'Adresse',
        'phone': 'Telefon',
        'hours': 'Öffnungszeiten',
        'services_available': 'Verfügbare Dienste',
        'emergency_housing': 'Notunterkunft',
        'important_note': 'Wichtiger Hinweis',
        'document_analysis': 'Dokumentenanalyse dient nur zu Informationszwecken - Keine Rechtsberatung',
        'need_more_help': 'Benötigen Sie Mehr Hilfe?',
        'common_questions': 'Häufige Fragen',
        'questions_about_document': 'Fragen zu diesem Dokument?',
        'go_to_chat': 'Gehen Sie zur Chat-Seite, um spezifische Fragen zu stellen',
        'upload_to_kb': 'Laden Sie dieses Dokument auf die Dokument-Upload-Seite hoch, um es zur LAISA-Wissensbasis hinzuzufügen',
        'consult_legal': 'Konsultieren Sie einen Rechtsexperten für spezifische Rechtsberatung',
        'document_summary': 'Dokumentzusammenfassung',
        'document_type': 'Dokumenttyp',
        'fields_to_fill': 'Auszufüllende Felder',
        'section_explanations': 'Abschnittserklärungen',
        'system_status': 'Systemstatus',
        'all_systems_operational': 'Alle Systeme betriebsbereit',
        'system_check_failed': 'Systemprüfung fehlgeschlagen',
        'check_system_status': 'Systemstatus Prüfen',
        'admin_information': 'Administrator-Informationen',
        'document_processing': 'Dokumentverarbeitung',
        'supported_formats': 'Unterstützte Formate',
        'security_note': 'Sicherheitshinweis',
        'language_support': 'Sprachunterstützung',
        'active_language': 'Aktive Sprache',
        'detected_language': 'Erkannte Sprache',
        'confidence': 'Vertrauen',
        'source': 'Quelle',
        'translation_active': 'Übersetzung: Aktiv',
        'translation_inactive': 'Übersetzung: Inaktiv (geringes Vertrauen)',
        'method': 'Methode',
        'script': 'Schrift',
        'indicators': 'Indikatoren',
        'statistical': 'Statistisch',
        'ai': 'KI',
        'alternative_candidates': 'Alternative Kandidaten',
        'current_session': 'Aktuelle Sitzung',
        'messages': 'Nachrichten',
        'language_english_default': 'Sprache: Englisch (Standard)',
        'housing_assistant': 'Wohnungsassistent',
        'multilingual_ai_assistant': 'Mehrsprachiger KI-Assistent für wohnungsbezogene Fragen',
        'reset_chat': 'Chat Zurücksetzen',
        'topics_help_with': 'Themen, Bei Denen Ich Helfen Kann',
        'tenant_rights': 'Mieterrechte und -pflichten',
        'landlord_obligations': 'Vermieterpflichten',
        'rental_agreements': 'Mietverträge und Mietvereinbarungen',
        'housing_regulations': 'Wohnungsbestimmungen',
        'eviction_processes': 'Räumungsverfahren',
        'property_management': 'Immobilienverwaltungsgesetze',
        'housing_discrimination': 'Wohnungsdiskriminierung',
        'under_construction': 'Im Aufbau',
        'documentation': 'Dokumentation',
        'contact': 'Kontakt',
        'document_help': 'Dokumenthilfe',
        'data_upload': 'Datenupload',
        'upload_analysis': 'Laden Sie ein Dokument hoch, um Analyse und Unterstützung zu erhalten',
        'pdf_only_supported': 'Nur PDF-Dateien werden unterstützt. Konvertieren Sie andere Formate zuerst zu PDF.',
        'failed_read_pdf': 'PDF konnte nicht gelesen werden',
        'try_different_method': 'Bitte versuchen Sie eine andere Methode, um Ihr Dokument in PDF-Format zu konvertieren',
        'analyzing_document': 'Dokument wird analysiert',
        'loading_document': 'Dokument wird geladen',
        'upload_pdf_to_start': 'Laden Sie ein PDF-Dokument hoch, um zu beginnen'
    },
    'nl': {
        # Common UI elements
        'upload_document': 'Document Uploaden',
        'drag_drop_files': 'Sleep je documenten hier naartoe',
        'processing': 'Verwerking',
        'loading': 'Laden',
        'upload_completed': 'Upload voltooid!',
        'upload_failed': 'Upload mislukt',
        'access_granted': 'Toegang Verleend',
        'access_denied': 'Toegang Geweigerd',
        'invalid_pin': 'Ongeldige PIN',
        'admin_only': 'ALLEEN BEHEERDER',
        'restricted_access': 'Beperkte Toegang',
        'enter_admin_pin': 'Beheerder PIN Invoeren',
        'contact_help': 'Contact en Overheidshuisvestingskantoren',
        'find_local_assistance': 'Lokale Huisvestingsondersteuning Vinden',
        'office_details': 'Kantoordetails',
        'address': 'Adres',
        'phone': 'Telefoon',
        'hours': 'Openingstijden',
        'services_available': 'Beschikbare Diensten',
        'emergency_housing': 'Noodhuisvesting',
        'important_note': 'Belangrijke Opmerking',
        'document_analysis': 'Documentanalyse is alleen voor informatieve doeleinden - Geen juridisch advies',
        'need_more_help': 'Meer Hulp Nodig?',
        'common_questions': 'Veelgestelde Vragen',
        'questions_about_document': 'Vragen over dit document?',
        'go_to_chat': 'Ga naar de Chat-pagina om specifieke vragen te stellen',
        'upload_to_kb': 'Upload dit document naar de Document Upload-pagina om het toe te voegen aan LAISA\'s kennisbank',
        'consult_legal': 'Raadpleeg een juridisch professional voor specifiek juridisch advies',
        'document_summary': 'Documentsamenvatting',
        'document_type': 'Documenttype',
        'fields_to_fill': 'Velden om In Te Vullen',
        'section_explanations': 'Sectie Uitleg',
        'system_status': 'Systeemstatus',
        'all_systems_operational': 'Alle systemen operationeel',
        'system_check_failed': 'Systeemcontrole mislukt',
        'check_system_status': 'Systeemstatus Controleren',
        'admin_information': 'Beheerder Informatie',
        'document_processing': 'Documentverwerking',
        'supported_formats': 'Ondersteunde Formaten',
        'security_note': 'Beveiligingsnotitie',
        'language_support': 'Taalondersteuning',
        'active_language': 'Actieve Taal',
        'detected_language': 'Gedetecteerde Taal',
        'confidence': 'Vertrouwen',
        'source': 'Bron',
        'translation_active': 'Vertaling: Actief',
        'translation_inactive': 'Vertaling: Inactief (laag vertrouwen)',
        'method': 'Methode',
        'script': 'Schrift',
        'indicators': 'Indicatoren',
        'statistical': 'Statistisch',
        'ai': 'AI',
        'alternative_candidates': 'Alternatieve kandidaten',
        'current_session': 'Huidige Sessie',
        'messages': 'Berichten',
        'language_english_default': 'Taal: Engels (Standaard)',
        'housing_assistant': 'Huisvestingsassistent',
        'multilingual_ai_assistant': 'Meertalige AI-assistent voor huisvestingsgerelateerde vragen',
        'reset_chat': 'Chat Resetten',
        'topics_help_with': 'Onderwerpen Waarmee Ik Kan Helpen',
        'tenant_rights': 'Huurderrechten en -verantwoordelijkheden',
        'landlord_obligations': 'Verhuurderverplichtingen',
        'rental_agreements': 'Huurovereenkomsten en -contracten',
        'housing_regulations': 'Huisvestingsreglementen',
        'eviction_processes': 'Uitzettingsprocedures',
        'property_management': 'Vastgoedbeheerswetten',
        'housing_discrimination': 'Huisvestingsdiscriminatie',
        'under_construction': 'In Aanbouw',
        'documentation': 'Documentatie',
        'contact': 'Contact',
        'document_help': 'Documenthulp',
        'data_upload': 'Data Upload',
        'upload_analysis': 'Upload een document om analyse en ondersteuning te krijgen',
        'pdf_only_supported': 'Alleen PDF-bestanden worden ondersteund. Converteer andere formaten eerst naar PDF.',
        'failed_read_pdf': 'PDF lezen mislukt',
        'try_different_method': 'Probeer een andere methode om je document naar PDF-formaat te converteren',
        'analyzing_document': 'Document analyseren',
        'loading_document': 'Document laden',
        'upload_pdf_to_start': 'Upload een PDF-document om te beginnen'
    }
}

def get_session_language_code() -> str:
    """Get the current session language code."""
    try:
        # Import here to avoid circular imports
        from core import get_session_language
        session_lang = get_session_language()
        return session_lang.get("language", "en")
    except:
        return "en"

def t(key: str, default: str = None) -> str:
    """
    Translate a text key to the current session language using dynamic translation.
    
    Args:
        key: Translation key
        default: Default text if translation not found
        
    Returns:
        Translated text or default/key if not found
    """
    try:
        current_language = get_session_language_code()
        
        # First try static translations for common keys (for performance)
        if current_language in GLOBAL_TRANSLATIONS:
            translation = GLOBAL_TRANSLATIONS[current_language].get(key)
            if translation:
                return translation
        
        # Fallback to English static translation
        english_text = None
        if 'en' in GLOBAL_TRANSLATIONS:
            english_text = GLOBAL_TRANSLATIONS['en'].get(key)
        
        # If no static translation found, create from key
        if not english_text:
            english_text = default if default else key.replace('_', ' ').title()
        
        # If current language is English, return the English text
        if current_language == 'en':
            return english_text
        
        # Use dynamic translation agent for non-English languages
        try:
            from core import translate_ui_text
            translated = translate_ui_text(english_text, current_language)
            return translated
        except ImportError:
            logger.warning("Dynamic translation not available - using static fallback")
            return english_text
        
    except Exception as e:
        logger.error(f"Translation error for key '{key}': {e}")
        return default if default else key.replace('_', ' ').title()

def get_available_languages() -> Dict[str, str]:
    """Get list of available languages (expanded for dynamic translation)."""
    return {
        'en': 'English',
        'es': 'Español',
        'fr': 'Français', 
        'de': 'Deutsch',
        'nl': 'Nederlands',
        'ar': 'العربية',
        'zh': '中文',
        'ja': '日本語',
        'ko': '한국어',
        'ru': 'Русский',
        'it': 'Italiano',
        'pt': 'Português',
        'tr': 'Türkçe',
        'pl': 'Polski',
        'sv': 'Svenska'
    }

def set_language_preference(language_code: str) -> bool:
    """Set user's language preference."""
    try:
        # Accept any language code for dynamic translation
        st.session_state.user_language_preference = language_code
        st.session_state.last_language_info = {
            "detected_language": language_code,
            "confidence": 1.0,
            "translated": language_code.lower() != "en",
            "source": "manual_selection"
        }
        logger.info(f"Language preference set to: {language_code}")
        return True
    except Exception as e:
        logger.error(f"Error setting language preference: {e}")
        return False

def display_language_selector():
    """Display a language selector widget."""
    try:
        current_lang = get_session_language_code()
        available_langs = get_available_languages()
        
        # Create selectbox
        selected_lang = st.selectbox(
            t('language_support'),
            options=list(available_langs.keys()),
            format_func=lambda x: f"{available_langs[x]} ({x.upper()})",
            index=list(available_langs.keys()).index(current_lang) if current_lang in available_langs else 0,
            key="global_language_selector"
        )
        
        # Update language if changed
        if selected_lang != current_lang:
            if set_language_preference(selected_lang):
                st.rerun()
                
    except Exception as e:
        logger.error(f"Error displaying language selector: {e}")