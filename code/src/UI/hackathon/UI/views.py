from django.core.cache import cache
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.db.models import Q
import json
import os
import re
import uuid
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .models import Transaction
from UI.AILayers.DataCollectionLayer import generate_risk_report
from UI.AILayers.ConfidenceCalculationLayer import compute_confidence_score
from UI.AILayers.RiskCalculationLayer import analyze_entity

def index(request):
    return render(request, "index.html")

@csrf_exempt
def handle_upload(request, folder="structured"):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_hash = uuid.uuid5(uuid.NAMESPACE_DNS, uploaded_file.name)
        cache_key = f"file_results_{file_hash}"
        
        # Check if cached results exist
        cached_results = cache.get(cache_key)
        if cached_results:
            return JsonResponse({"results": cached_results}, status=200)

        upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        extracted_transactions = process_file(file_path, folder)
        if not extracted_transactions:
            return JsonResponse({'error': 'No transactions found in file'}, status=400)
        
        results = []
        for txn in extracted_transactions:
            transaction = Transaction.objects.create(
                sender=txn.get('sender', None),
                receiver=txn.get('receiver', None),
                entity_type=txn.get('entity_type', ["Unknown"]),
                risk_score=txn.get('risk_score', 0.5),
                supporting_evidence=txn.get('supporting_evidence', ["No evidence available"]),
                confidence_score=txn.get('confidence_score', 0.5),
                reason=txn.get('reason', "No risk detected."),
                full_text=txn['full_text']
            )

            dataReportSender = generate_risk_report(transaction.sender)
            dataReportReceiver = generate_risk_report(transaction.receiver)
            riskSender = analyze_entity(dataReportSender)
            riskReceiver = analyze_entity(dataReportReceiver)
            
            if riskSender["risk_score"] > riskReceiver["risk_score"]:
                riskScore = riskSender["risk_score"]
                justification = riskSender["justification"]
                confidence = compute_confidence_score(riskSender["risk_score"], riskSender["justification"])
            else:
                riskScore = riskReceiver["risk_score"]
                justification = riskReceiver["justification"]
                confidence = compute_confidence_score(riskReceiver["risk_score"], riskReceiver["justification"])
            
            results.append({
                "Transaction ID": str(transaction.transaction_id),
                "Sender": transaction.sender,
                "Receiver": transaction.receiver,
                "Entity Type": transaction.entity_type,
                "Risk Score": riskScore,
                "Supporting Evidence": justification,
                "Confidence Score": confidence,
                "Reason": transaction.reason,
            })
        
        # Store in cache
        cache.set(cache_key, results, timeout=86400)  # Cache for 24 hours
        return JsonResponse({"results": results}, status=200)
    
    return JsonResponse({'error': 'Upload failed'}, status=400)


def process_file(file_path, folder):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []

    transactions = content.split('---')
    extracted_data = []
    for transaction in transactions:
        sender_match = re.search(r"Sender:\s*- Name:\s*\"(.*?)\"", transaction)
        receiver_match = re.search(r"Receiver:\s*- Name:\s*\"(.*?)\"", transaction)
        if folder == "structured" and sender_match and receiver_match:
            extracted_data.append({
                "sender": sender_match.group(1),
                "receiver": receiver_match.group(1),
                "entity_type": ["Corporation"],
                "risk_score": 0.9,
                "supporting_evidence": ["Company database"],
                "confidence_score": 0.85,
                "reason": f"{sender_match.group(1)} and {receiver_match.group(1)} appear in company records.",
                "full_text": transaction.strip()
            })
        else:
            extracted_data.append({
                "sender": None,
                "receiver": None,
                "entity_type": ["Unknown"],
                "risk_score": 0.5,
                "supporting_evidence": ["No clear match"],
                "confidence_score": 0.5,
                "reason": "Could not extract structured sender/receiver data.",
                "full_text": transaction.strip()
            })
    return extracted_data

@csrf_exempt
def search_entity(request):
    if request.method == 'GET':
        entity_name = request.GET.get('entity', '').strip()
        if not entity_name: 
            return HttpResponse("Entity name required", status=400, content_type="text/plain")
        
        cache_key = f"entity_report_{entity_name}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return HttpResponse(cached_result, content_type="text/plain")
        
        result = generate_risk_report(entity_name)
        cache.set(cache_key, result, timeout=86400)  # Cache for 24 hours
        return HttpResponse(result, content_type="text/plain")
    
    return HttpResponse("Invalid request", status=400, content_type="text/plain")
