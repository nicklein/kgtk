#! /bin/bash

source common.sh

# ==============================================================================
echo -e "\nCount the entities in the node1 column in ${DATADIR}/aliases.en.${SORTED_KGTK} and lift labels."
kgtk ${KGTK_FLAGS} \
     unique ${VERBOSE} --use-mgzip=${USE_MGZIP} --presorted \
     --input-file ${DATADIR}/aliases.en.${SORTED_KGTK} \
     --column node1 \
     --label node1-entity-count \
     / lift ${VERBOSE} --use-mgzip=${USE_MGZIP} \
     --label-file ${DATADIR}/labels.en.${SORTED_KGTK} \
     --output-file ${DATADIR}/aliases.en.node1.entity.counts.${SORTED_KGTK} \
     --columns-to-lift node1 \
     --input-file-is-presorted \
     --label-file-is-presorted \
     |& tee ${LOGDIR}/aliases.en.node1.entity.counts.log

echo -e "\nCount the entities in the node1 column in ${DATADIR}/descriptions.en.${SORTED_KGTK} and lift labels."
kgtk ${KGTK_FLAGS} \
     unique ${VERBOSE} --use-mgzip=${USE_MGZIP} --presorted \
     --input-file ${DATADIR}/descriptions.en.${SORTED_KGTK} \
     --column node1 \
     --label node1-entity-count \
     / lift ${VERBOSE} --use-mgzip=${USE_MGZIP} \
     --label-file ${DATADIR}/labels.en.${SORTED_KGTK} \
     --output-file ${DATADIR}/descriptions.en.node1.entity.counts.${SORTED_KGTK} \
     --columns-to-lift node1 \
     --input-file-is-presorted \
     --label-file-is-presorted \
     |& tee ${LOGDIR}/descriptions.en.node1.entity.counts.log

echo -e "\nCount the entities in the node1 column in ${DATADIR}/labels.${SORTED_KGTK} and lift labels."
kgtk ${KGTK_FLAGS} \
     unique ${VERBOSE} --use-mgzip=${USE_MGZIP} --presorted \
     --input-file ${DATADIR}/labels.en.${SORTED_KGTK} \
     --column node1 \
     --label node1-entity-count \
     / lift ${VERBOSE} --use-mgzip=${USE_MGZIP} \
     --label-file ${DATADIR}/labels.en.${SORTED_KGTK} \
     --output-file ${DATADIR}/labels.en.node1.entity.counts.${SORTED_KGTK} \
     --columns-to-lift node1 \
     --input-file-is-presorted \
     --label-file-is-presorted \
     |& tee ${LOGDIR}/labels.en.node1.entity.counts.log

echo -e "\nCount the entities in the node1 column in ${DATADIR}/sitelinks.en.${SORTED_KGTK} and lift labels."
kgtk ${KGTK_FLAGS} \
     unique ${VERBOSE} --use-mgzip=${USE_MGZIP} --presorted \
     --input-file ${DATADIR}/sitelinks.en.${SORTED_KGTK} \
     --column node1 \
     --label node1-entity-count \
     / lift ${VERBOSE} --use-mgzip=${USE_MGZIP} \
     --label-file ${DATADIR}/labels.en.${SORTED_KGTK} \
     --output-file ${DATADIR}/sitelinks.en.node1.entity.counts.${SORTED_KGTK} \
     --columns-to-lift node1 \
     --input-file-is-presorted \
     --label-file-is-presorted \
     |& tee ${LOGDIR}/sitelinks.en.node1.entity.counts.log