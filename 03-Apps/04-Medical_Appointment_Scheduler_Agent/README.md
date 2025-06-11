# Medical Appointment Scheduler Agent

## Description
This agent schedules medical appointments for patients. It integrates with calendar and insurance APIs, performs symptom triage for urgency routing, and automates follow-up reminders.

## Pseudocode
```
FUNCTION handle_appointment_request(user_message, patient_data, conversation_history)
  intent = IDENTIFY_INTENT(user_message)
  entities = EXTRACT_ENTITIES(user_message) // e.g., doctor_name, preferred_date, symptoms

  IF intent == "schedule_appointment" THEN
    // Symptom Triage
    IF entities.symptoms IS NOT NULL THEN
      urgency = TRIAGE_SYMPTOMS(entities.symptoms)
      IF urgency == "emergency" THEN
        response = "Your symptoms sound serious. Please contact emergency services immediately."
        RETURN response
      ENDIF
    ELSE
      urgency = "routine"
    ENDIF

    // Insurance Check
    IF patient_data.insurance_details IS NOT NULL THEN
      is_covered = CHECK_INSURANCE_COVERAGE(entities.doctor_name, entities.appointment_type, patient_data.insurance_details)
      IF NOT is_covered THEN
        response = "Dr. " + entities.doctor_name + " may not be covered by your insurance for this type of appointment. Would you like to proceed or find another doctor?"
        SET_PENDING_ACTION(conversation_history, "insurance_confirmation")
        RETURN response
      ENDIF
    ENDIF

    // Availability Check & Booking
    available_slots = GET_DOCTOR_AVAILABILITY(entities.doctor_name, entities.preferred_date, urgency)
    IF available_slots IS EMPTY THEN
      response = "Dr. " + entities.doctor_name + " has no availability around " + entities.preferred_date + ". Would you like to see other dates or doctors?"
      // Offer alternative slots or doctors
    ELSE
      chosen_slot = PRESENT_AND_CONFIRM_SLOT(available_slots, user_message)
      IF chosen_slot IS CONFIRMED THEN
        appointment_id = BOOK_APPOINTMENT(patient_data, entities.doctor_name, chosen_slot)
        ADD_TO_CALENDAR(patient_data.calendar_api_token, appointment_id, chosen_slot, entities.doctor_name)
        SCHEDULE_FOLLOW_UP_REMINDER(patient_data, appointment_id, chosen_slot)
        response = "Your appointment with Dr. " + entities.doctor_name + " on " + chosen_slot.date + " at " + chosen_slot.time + " is confirmed. (ID: " + appointment_id + ")"
      ELSE
        response = "Okay, let me know if you'd like to try other options."
      ENDIF
    ENDIF
  ELSEIF intent == "cancel_appointment" THEN
    appointment_id = entities.appointment_id
    success = CANCEL_APPOINTMENT(appointment_id, patient_data)
    IF success THEN
      REMOVE_FROM_CALENDAR(patient_data.calendar_api_token, appointment_id)
      response = "Your appointment (ID: " + appointment_id + ") has been cancelled."
    ELSE
      response = "Sorry, I couldn't cancel appointment " + appointment_id + ". Please contact the clinic directly."
    ENDIF
  ELSEIF intent == "reschedule_appointment" THEN
    // Similar logic to schedule_appointment, but first cancel existing one
    response = "Okay, let's find a new time for your appointment."
  ELSE
    response = "How can I help you with medical appointments today?"
  ENDIF

  ADD_TO_CONVERSATION_HISTORY(user_message, response)
  RETURN response
END FUNCTION

FUNCTION TRIAGE_SYMPTOMS(symptoms_list)
  // Rule-based or ML model for symptom severity
  IF ANY_SEVERE_SYMPTOM(symptoms_list) THEN
    RETURN "emergency"
  ELSEIF ANY_MODERATE_SYMPTOM(symptoms_list) THEN
    RETURN "urgent"
  ELSE
    RETURN "routine"
  ENDIF
END FUNCTION

FUNCTION CHECK_INSURANCE_COVERAGE(doctor, appointment_type, insurance_plan)
  // API call to insurance provider or internal database
  is_in_network = CALL_INSURANCE_API_CHECK_DOCTOR(doctor, insurance_plan)
  is_service_covered = CALL_INSURANCE_API_CHECK_SERVICE(appointment_type, insurance_plan)
  RETURN is_in_network AND is_service_covered
END FUNCTION

FUNCTION GET_DOCTOR_AVAILABILITY(doctor, preferred_date, urgency_level)
  // API call to clinic's scheduling system
  slots = CALL_SCHEDULING_API_GET_SLOTS(doctor, preferred_date, urgency_level)
  RETURN slots
END FUNCTION

FUNCTION BOOK_APPOINTMENT(patient, doctor, slot)
  // API call to clinic's scheduling system
  appointment_id = CALL_SCHEDULING_API_BOOK(patient.id, doctor, slot)
  RETURN appointment_id
END FUNCTION

FUNCTION SCHEDULE_FOLLOW_UP_REMINDER(patient, appointment_id, slot)
  // Add reminder to a queue or scheduling service
  reminder_time = slot.datetime - 1_DAY
  CREATE_REMINDER_EVENT(patient.contact_info, appointment_id, reminder_time, "Upcoming appointment reminder")
END FUNCTION
```
