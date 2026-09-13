package org.omnaest.utils.neuralnetwork;

import org.junit.jupiter.api.Test;
import org.omnaest.utils.style.StyleProfile;
import org.omnaest.utils.style.sourcetext.SourceGuard;

/**
 * Mechanically enforces this workspace's Java package structure guideline (see
 * {@code .claude/guidelines/java-package-structure.md}) against {@code CommonsNeuralNetwork} via
 * {@code CommonsStyleSupport}'s LIBRARY profile (plan-188).
 * <p>
 * One {@code @Test} method per check, so a project opts out of one check by deleting one line. This class is
 * test-only structural-assertion infrastructure with no main-source counterpart, so it is exempt from the
 * test-mirror rule (P13) and legitimately sits at the context root.
 * <p>
 * <b>plan-216:</b> this project adopted {@code CommonsStyleSupport} when the rule surface was smaller and had
 * drifted to 10 of the module's 16 enforced checks. Brought up to the full enforced surface here. Two checks are
 * deliberately excluded, both documented measurement-only in their own source rather than shipped enforcement:
 * {@code StyleProfile.internalPackagesAreAccessedOnlyFromTheirDirectParentPackage()} and
 * {@code SourceGuard.noInternalReferencesFromOutsideTheirDirectParentPackage()}.
 */
class PackageStructureTest
{

    private static final StyleProfile PROFILE = StyleProfile.library("org.omnaest.utils.neuralnetwork");

    @Test
    void singleEntryPointAtContextRoot()
    {
        PROFILE.singleEntryPointAtContextRoot()
               .check(PROFILE.mainClasses());
    }

    @Test
    void entryPointIsInterfaceOrUtilsFactory()
    {
        PROFILE.entryPointIsInterfaceOrUtilsFactory()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noHorizontalLayerPackages()
    {
        PROFILE.noHorizontalLayerPackages()
               .check(PROFILE.mainClasses());
    }

    @Test
    void internalPackagesAreAccessedOnlyFromWithinTheirOwnSubtree()
    {
        PROFILE.internalPackagesAreAccessedOnlyFromWithinTheirOwnSubtree()
               .check(PROFILE.mainClasses());
    }

    @Test
    void repositoryTypesLiveInInternalRepository()
    {
        PROFILE.repositoryTypesLiveInInternalRepository()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noDtoTypesOutsideInternal()
    {
        PROFILE.noDtoTypesOutsideInternal()
               .check(PROFILE.mainClasses());
    }

    @Test
    void internalSubPackagesAreRoleNamed()
    {
        PROFILE.internalSubPackagesAreRoleNamed()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noInternalTypeOnAPublicApiSurface()
    {
        PROFILE.noInternalTypeOnAPublicApiSurface()
               .check(PROFILE.mainClasses());
    }

    @Test
    void boundedContextsAreDiscovered()
    {
        PROFILE.boundedContextsAreDiscovered()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noContextDependsOnAnAdapter()
    {
        PROFILE.noContextDependsOnAnAdapter()
               .check(PROFILE.mainClasses());
    }

    @Test
    void adapterWireTypesLiveInTheirChannelDomain()
    {
        PROFILE.adapterWireTypesLiveInTheirChannelDomain()
               .check(PROFILE.mainClasses());
    }

    @Test
    void sharedTypesAreUsedByAtLeastTwoContexts()
    {
        PROFILE.sharedTypesAreUsedByAtLeastTwoContexts()
               .check(PROFILE.mainClasses());
    }

    @Test
    void utilsPackagesDoNotReachIntoDomain()
    {
        PROFILE.utilsPackagesDoNotReachIntoDomain()
               .check(PROFILE.mainClasses());
    }

    @Test
    void utilsPackagesDoNotDuplicateCommonsTypes()
    {
        PROFILE.utilsPackagesDoNotDuplicateCommonsTypes()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noInternalReferencesFromOutsideTheirOwnSubtree()
    {
        SourceGuard.of()
                   .noInternalReferencesFromOutsideTheirOwnSubtree()
                   .verify();
    }

    @Test
    void testsMirrorTheirSubjectPackage()
    {
        SourceGuard.of()
                   .testsMirrorTheirSubjectPackage()
                   .verify();
    }

}
