"""Built-in GraphGuard quality contracts."""
from __future__ import annotations

from hckg_enrich.guard.contracts.circular_dependency import CircularDependencyContract
from hckg_enrich.guard.contracts.data_asset_ownership import DataAssetOwnershipContract
from hckg_enrich.guard.contracts.entity_deduplication import EntityDeduplicationContract
from hckg_enrich.guard.contracts.org_hierarchy import OrgHierarchyContract
from hckg_enrich.guard.contracts.person_role_consistency import PersonRoleConsistencyContract
from hckg_enrich.guard.contracts.plausibility import PlausibilityContract
from hckg_enrich.guard.contracts.relationship_semantics import RelationshipTypeSemanticsContract
from hckg_enrich.guard.contracts.system_ownership import SystemOwnershipContract
from hckg_enrich.guard.contracts.vendor_relationship import VendorRelationshipContract

__all__ = [
    "CircularDependencyContract",
    "DataAssetOwnershipContract",
    "EntityDeduplicationContract",
    "OrgHierarchyContract",
    "PersonRoleConsistencyContract",
    "PlausibilityContract",
    "RelationshipTypeSemanticsContract",
    "SystemOwnershipContract",
    "VendorRelationshipContract",
]
